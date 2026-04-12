"""Tests for the LLM client wrapper.

No real LLM calls happen in this file. Backends are tested via the
FakeLLMClient and inline test doubles. The real Anthropic/OpenAI
clients are exercised lightly through the factory (instantiation only,
no network).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from convgen.llm.client import (
    DiskCache,
    FakeLLMClient,
    LLMClient,
    LLMError,
    LLMParseError,
    _strictify,
    make_client,
)


class _Plan(BaseModel):
    """Schema used by structured-output tests."""

    intent: str
    needs_clarification: bool
    expected_turns: int = Field(ge=1, le=20)


# ---------------------- DiskCache ----------------------


class TestDiskCache:
    def test_set_and_get(self, tmp_path: Path):
        cache = DiskCache(tmp_path)
        cache.set({"k": "v"}, {"answer": 42})
        assert cache.get({"k": "v"}) == {"answer": 42}

    def test_miss_returns_none(self, tmp_path: Path):
        cache = DiskCache(tmp_path)
        assert cache.get({"never": "set"}) is None

    def test_keys_are_order_independent(self, tmp_path: Path):
        cache = DiskCache(tmp_path)
        cache.set({"a": 1, "b": 2}, "stored")
        assert cache.get({"b": 2, "a": 1}) == "stored"

    def test_different_payloads_different_entries(self, tmp_path: Path):
        cache = DiskCache(tmp_path)
        cache.set({"k": "v1"}, "one")
        cache.set({"k": "v2"}, "two")
        assert cache.get({"k": "v1"}) == "one"
        assert cache.get({"k": "v2"}) == "two"

    def test_corrupted_cache_file_returns_none(self, tmp_path: Path):
        cache = DiskCache(tmp_path)
        cache.set({"k": "v"}, "x")
        for cache_file in tmp_path.iterdir():
            cache_file.write_text("not json", encoding="utf-8")
        assert cache.get({"k": "v"}) is None


# ---------------------- FakeLLMClient — text ----------------------


class TestFakeClientText:
    def test_default_response(self):
        client = FakeLLMClient(default_text="hello")
        assert client.complete("anything") == "hello"

    def test_exact_match(self):
        client = FakeLLMClient(text_responses={"ping": "pong"})
        assert client.complete("ping") == "pong"

    def test_substring_match(self):
        client = FakeLLMClient(text_responses={"weather": "sunny"})
        assert client.complete("Tell me the weather please") == "sunny"

    def test_calls_are_recorded(self):
        client = FakeLLMClient()
        client.complete("first")
        client.complete("second", system="be brief")
        assert len(client.calls) == 2
        assert client.calls[0]["prompt"] == "first"
        assert client.calls[1]["system"] == "be brief"


# ---------------------- FakeLLMClient — structured ----------------------


class TestFakeClientStructured:
    def test_returns_validated_pydantic_instance(self):
        client = FakeLLMClient(
            default_structured={
                "intent": "book a hotel",
                "needs_clarification": False,
                "expected_turns": 5,
            }
        )
        plan = client.complete_structured("plan something", _Plan)
        assert isinstance(plan, _Plan)
        assert plan.intent == "book a hotel"
        assert plan.expected_turns == 5

    def test_substring_match_for_structured(self):
        good = {
            "intent": "i",
            "needs_clarification": True,
            "expected_turns": 3,
        }
        client = FakeLLMClient(structured_responses={"hotels": good})
        plan = client.complete_structured("plan a hotels trip", _Plan)
        assert plan.expected_turns == 3
        assert plan.needs_clarification is True

    def test_validation_failure_raises_after_retries(self):
        # 999 violates Field(le=20). Pydantic will reject it on every
        # attempt, so the repair loop should exhaust and raise.
        client = FakeLLMClient(
            default_structured={
                "intent": "x",
                "needs_clarification": False,
                "expected_turns": 999,
            }
        )
        with pytest.raises(LLMParseError):
            client.complete_structured("plan", _Plan, max_repair_attempts=1)

    def test_structured_call_records_strict_flag(self):
        client = FakeLLMClient(
            default_structured={
                "intent": "x",
                "needs_clarification": False,
                "expected_turns": 2,
            }
        )
        client.complete_structured("plan", _Plan, strict_schema=False)
        assert client.calls[-1]["strict"] is False


# ---------------------- Repair loop ----------------------


class _StatefulFake(LLMClient):
    """Test double that returns a queued sequence of structured
    responses, used to exercise the repair retry path."""

    def __init__(self, queue: list[dict]) -> None:
        super().__init__(model="stateful-fake", cache=None)
        self.queue = list(queue)
        self.attempts = 0

    @property
    def provider_name(self) -> str:
        return "stateful-fake"

    def _raw_complete(self, prompt, system, temperature):
        return ""

    def _raw_structured(self, prompt, system, schema, temperature, strict=True):
        self.attempts += 1
        return self.queue.pop(0)


class TestStructuredRepairLoop:
    def test_first_invalid_then_valid_succeeds(self):
        client = _StatefulFake(
            queue=[
                {"intent": "x"},  # invalid: missing fields
                {
                    "intent": "x",
                    "needs_clarification": False,
                    "expected_turns": 4,
                },
            ]
        )
        plan = client.complete_structured("p", _Plan, max_repair_attempts=2)
        assert plan.expected_turns == 4
        assert client.attempts == 2

    def test_all_invalid_raises_after_exhausting_attempts(self):
        client = _StatefulFake(
            queue=[
                {"bad": 1},
                {"bad": 2},
                {"bad": 3},
            ]
        )
        with pytest.raises(LLMParseError):
            client.complete_structured("p", _Plan, max_repair_attempts=2)
        assert client.attempts == 3  # initial + 2 repairs


# ---------------------- Caching integration ----------------------


class _CountingFake(LLMClient):
    """Test double that counts how many raw calls were made."""

    def __init__(self, cache: DiskCache | None = None) -> None:
        super().__init__(model="counting", cache=cache)
        self.text_calls = 0
        self.structured_calls = 0

    @property
    def provider_name(self) -> str:
        return "counting"

    def _raw_complete(self, prompt, system, temperature):
        self.text_calls += 1
        return f"answer to: {prompt}"

    def _raw_structured(self, prompt, system, schema, temperature, strict=True):
        self.structured_calls += 1
        return {
            "intent": "x",
            "needs_clarification": False,
            "expected_turns": 2,
        }


class TestCachingIntegration:
    def test_text_cache_hit(self, tmp_path: Path):
        client = _CountingFake(cache=DiskCache(tmp_path))
        client.complete("hi")
        client.complete("hi")
        assert client.text_calls == 1

    def test_text_cache_miss_on_different_prompt(self, tmp_path: Path):
        client = _CountingFake(cache=DiskCache(tmp_path))
        client.complete("hi")
        client.complete("bye")
        assert client.text_calls == 2

    def test_structured_cache_hit(self, tmp_path: Path):
        client = _CountingFake(cache=DiskCache(tmp_path))
        client.complete_structured("plan", _Plan)
        client.complete_structured("plan", _Plan)
        assert client.structured_calls == 1

    def test_structured_cache_miss_when_strict_schema_changes(
        self, tmp_path: Path
    ):
        client = _CountingFake(cache=DiskCache(tmp_path))
        client.complete_structured("plan", _Plan, strict_schema=True)
        client.complete_structured("plan", _Plan, strict_schema=False)
        assert client.structured_calls == 2

    def test_temperature_change_busts_cache(self, tmp_path: Path):
        client = _CountingFake(cache=DiskCache(tmp_path))
        client.complete("hi", temperature=0.0)
        client.complete("hi", temperature=0.5)
        assert client.text_calls == 2

    def test_no_cache_means_no_skipping(self):
        client = _CountingFake(cache=None)
        client.complete("hi")
        client.complete("hi")
        assert client.text_calls == 2


# ---------------------- _strictify ----------------------


class TestStrictify:
    def test_adds_additional_properties_false(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        out = _strictify(schema)
        assert out["additionalProperties"] is False

    def test_adds_required_for_all_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
        }
        out = _strictify(schema)
        assert set(out["required"]) == {"a", "b"}

    def test_replaces_partial_required_openai_strict(self):
        """Pydantic emits partial ``required`` for defaulted fields; OpenAI
        strict mode needs every property key listed."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            "required": ["a"],
        }
        out = _strictify(schema)
        assert set(out["required"]) == {"a", "b"}

    def test_recurses_into_nested_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                }
            },
        }
        out = _strictify(schema)
        assert out["properties"]["inner"]["additionalProperties"] is False

    def test_does_not_mutate_input(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        before = json.dumps(schema, sort_keys=True)
        _strictify(schema)
        after = json.dumps(schema, sort_keys=True)
        assert before == after

    def test_loose_schema_not_run_through_strictify(self):
        """OpenAI uses raw Pydantic JSON Schema when ``strict_schema`` is false
        (Assistant path): no ``required`` backfill on the root object."""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
        }
        loose = json.loads(json.dumps(schema))
        strict = _strictify(json.loads(json.dumps(schema)))
        assert "required" not in loose
        assert set(strict["required"]) == {"a", "b"}


class _StrictRecording(LLMClient):
    """Records the ``strict`` flag passed to ``_raw_structured``."""

    def __init__(self) -> None:
        super().__init__(model="strict-rec", cache=None)
        self.last_strict: bool | None = None

    @property
    def provider_name(self) -> str:
        return "strict-rec"

    def _raw_complete(self, prompt, system, temperature):
        return ""

    def _raw_structured(self, prompt, system, schema, temperature, strict=True):
        self.last_strict = strict
        return {
            "intent": "ok",
            "needs_clarification": False,
            "expected_turns": 2,
        }


class TestCompleteStructuredStrictPassthrough:
    def test_passes_strict_schema_to_raw(self):
        client = _StrictRecording()
        client.complete_structured("p", _Plan, strict_schema=False)
        assert client.last_strict is False
        client.complete_structured("p", _Plan, strict_schema=True)
        assert client.last_strict is True


# ---------------------- Factory ----------------------


class TestMakeClient:
    def test_no_provider_no_keys_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CONVGEN_LLM_PROVIDER", raising=False)
        with pytest.raises(LLMError):
            make_client(cache_dir=None)

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "stub")
        with pytest.raises(LLMError):
            make_client(provider="wibble", cache_dir=None)

    def test_anthropic_picked_when_only_anthropic_key_set(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "stub-key-not-used")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CONVGEN_LLM_PROVIDER", raising=False)
        client = make_client(cache_dir=tmp_path)
        assert client.provider_name == "anthropic"

    def test_openai_picked_when_only_openai_key_set(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "stub-key-not-used")
        monkeypatch.delenv("CONVGEN_LLM_PROVIDER", raising=False)
        client = make_client(cache_dir=tmp_path)
        assert client.provider_name == "openai"

    def test_explicit_provider_wins_over_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "stub")
        monkeypatch.setenv("OPENAI_API_KEY", "stub")
        monkeypatch.setenv("CONVGEN_LLM_PROVIDER", "anthropic")
        client = make_client(provider="openai", cache_dir=tmp_path)
        assert client.provider_name == "openai"
