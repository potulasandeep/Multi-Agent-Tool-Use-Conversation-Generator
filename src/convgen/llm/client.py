"""
Provider-agnostic LLM client used by every agent and the judge.

Public surface
--------------
- `LLMClient` (ABC): the interface the rest of the system depends on.
  Two methods: `complete` (free-text) and `complete_structured`
  (Pydantic-validated JSON).
- `AnthropicClient`, `OpenAIClient`: real backends. Anthropic is the
  default for this build; OpenAI is supported for portability.
- `FakeLLMClient`: returns canned responses for tests, matching by
  exact prompt or substring. The structured variant returns a dict
  that the base class validates against the requested Pydantic model,
  so tests still exercise the validation/repair path.
- `DiskCache`: hashes (provider, model, prompt, schema, temperature, strict)
  into a filename and stores the response as JSON. Survives process
  restarts. Cuts iterative-development cost dramatically.
- `make_client()`: factory that picks a provider based on env vars.

Design notes
------------
Structured output is implemented per-provider:
  Anthropic: forced tool-use with the requested schema as input_schema.
  OpenAI: response_format JSON schema; ``strict_schema`` is per call (Planner/Judge
  use strict mode; Assistant uses non-strict for open ``tool_arguments`` dicts).
The public method `complete_structured` accepts a Pydantic model class
so the calling code gets full type safety and the LLM gets a real
JSON Schema, both from one definition.

When structured output fails Pydantic validation, the client retries
with the validation error injected back into the prompt
("your previous response failed validation: ..."). After
`max_repair_attempts` retries, raises LLMParseError. Agents in Step 8
can therefore assume any returned model is valid.

Cache keys include temperature so that the User agent's varied
sampling doesn't collide with the Planner/Assistant/Judge's
temperature-0 calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_CACHE_DIR = Path(".llm_cache")


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class LLMError(Exception):
    """Generic LLM client failure (network, config, missing SDK, etc.)."""


class LLMParseError(LLMError):
    """Structured output failed to parse or validate after all repair
    attempts. Raised by `complete_structured` only."""


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------


class DiskCache:
    """Tiny content-addressed cache: one JSON file per request hash.

    Keys are produced by canonicalizing the request payload (sorted
    JSON) and SHA-256 hashing it. Values are arbitrary JSON-serializable
    objects. Corrupted cache files return None on read.
    """

    def __init__(self, directory: Path | str = DEFAULT_CACHE_DIR) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, payload: dict) -> str:
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    def get(self, payload: dict) -> Any | None:
        path = self.dir / f"{self._key(payload)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupted cache file: %s", path)
            return None

    def set(self, payload: dict, value: Any) -> None:
        path = self.dir / f"{self._key(payload)}.json"
        path.write_text(json.dumps(value, default=str), encoding="utf-8")


# --------------------------------------------------------------------------
# Base client
# --------------------------------------------------------------------------


class LLMClient(ABC):
    """Abstract base for all LLM clients.

    Subclasses implement two private methods (`_raw_complete` and
    `_raw_structured`); the public methods (`complete`,
    `complete_structured`) handle caching and the structured-output
    repair loop in one place.
    """

    def __init__(
        self,
        model: str,
        cache: DiskCache | None = None,
    ) -> None:
        self.model = model
        self.cache = cache

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def _raw_complete(
        self, prompt: str, system: str | None, temperature: float
    ) -> str: ...

    @abstractmethod
    def _raw_structured(
        self,
        prompt: str,
        system: str | None,
        schema: dict,
        temperature: float,
        strict: bool = True,
    ) -> dict: ...

    # -- public API ----------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "kind": "text",
            "provider": self.provider_name,
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
        }
        if self.cache is not None:
            cached = self.cache.get(payload)
            if cached is not None:
                return cached["text"]
        text = self._raw_complete(prompt, system, temperature)
        if self.cache is not None:
            self.cache.set(payload, {"text": text})
        return text

    def complete_structured(
        self,
        prompt: str,
        schema_model: type[T],
        system: str | None = None,
        temperature: float = 0.0,
        max_repair_attempts: int = 1,
        strict_schema: bool = True,
    ) -> T:
        """Return a validated instance of `schema_model`.

        On a validation failure, retries up to `max_repair_attempts`
        times with the error injected into the prompt. Raises
        LLMParseError if every attempt fails.

        ``strict_schema`` is passed to the provider (OpenAI: JSON schema
        ``strict`` flag and whether to run ``_strictify``). Closed schemas
        (Planner, Judge) should use the default ``True``; open dict fields
        (Assistant ``tool_arguments``) need ``False`` on OpenAI.
        """
        schema = schema_model.model_json_schema()
        payload = {
            "kind": "structured",
            "provider": self.provider_name,
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "schema": schema,
            "temperature": temperature,
            "strict": strict_schema,
        }
        if self.cache is not None:
            cached = self.cache.get(payload)
            if cached is not None:
                try:
                    return schema_model.model_validate(cached)
                except ValidationError:
                    logger.warning(
                        "Cached structured response failed validation; "
                        "recomputing"
                    )

        last_error: Exception | None = None
        attempt_prompt = prompt
        for attempt in range(max_repair_attempts + 1):
            try:
                raw = self._raw_structured(
                    attempt_prompt,
                    system,
                    schema,
                    temperature,
                    strict_schema,
                )
                parsed = schema_model.model_validate(raw)
                if self.cache is not None:
                    self.cache.set(payload, raw)
                return parsed
            except ValidationError as error:
                last_error = error
                logger.warning(
                    "Structured parse failed on attempt %d/%d: %s",
                    attempt + 1,
                    max_repair_attempts + 1,
                    error,
                )
                attempt_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed validation:\n{error}\n\n"
                    f"Return a valid JSON object matching the schema."
                )
        raise LLMParseError(
            f"Failed to parse structured output after "
            f"{max_repair_attempts + 1} attempts: {last_error}"
        )


# --------------------------------------------------------------------------
# Anthropic backend
# --------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    """Anthropic backend.

    Structured output uses forced tool-use: a single tool named
    `respond` is registered whose input_schema IS the requested schema,
    and tool_choice forces the model to call it. The tool input block
    is the structured response.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        super().__init__(model=model, cache=cache)
        try:
            import anthropic
        except ImportError as error:
            raise LLMError("anthropic SDK not installed") from error
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _raw_complete(
        self, prompt: str, system: str | None, temperature: float
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise LLMError("Anthropic returned no text content")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _raw_structured(
        self,
        prompt: str,
        system: str | None,
        schema: dict,
        temperature: float,
        strict: bool = True,
    ) -> dict:
        _ = strict  # OpenAI-only; Anthropic tool input is always schema-bound.
        tool = {
            "name": "respond",
            "description": "Respond with structured data matching the schema.",
            "input_schema": schema,
        }
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": temperature,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "respond"},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        for block in resp.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "respond"
            ):
                return dict(block.input)
        raise LLMError("Anthropic returned no tool_use block")


# --------------------------------------------------------------------------
# OpenAI backend
# --------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    """OpenAI backend.

    Structured output uses ``response_format`` JSON schema. When
    ``strict`` is true, the schema is passed through ``_strictify`` and
    ``json_schema.strict`` is enabled. Callers with open object fields
    (e.g. Assistant ``tool_arguments``) pass ``strict=False`` so the
    Pydantic schema is sent as-is and strict mode stays off.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        super().__init__(model=model, cache=cache)
        try:
            import openai
        except ImportError as error:
            raise LLMError("openai SDK not installed") from error
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _raw_complete(
        self, prompt: str, system: str | None, temperature: float
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _raw_structured(
        self,
        prompt: str,
        system: str | None,
        schema: dict,
        temperature: float,
        strict: bool = True,
    ) -> dict:
        final_schema = _strictify(schema) if strict else schema
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": final_schema,
                    "strict": strict,
                },
            },
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)


def _strictify(schema: dict) -> dict:
    """Return a copy of `schema` with the OpenAI strict-mode additions:
    `additionalProperties: false` and `required: [all properties]` on
    every object level. Does not mutate the input."""
    out = json.loads(json.dumps(schema))  # cheap deep copy

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
                if "properties" in node:
                    # OpenAI strict mode: `required` must list every key in
                    # `properties`. Pydantic omits defaulted fields from
                    # `required`; setdefault would leave that incomplete schema.
                    node["required"] = list(node["properties"].keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(out)
    return out


# --------------------------------------------------------------------------
# Fake client for tests
# --------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    """In-memory test double.

    Match strategy for both text and structured calls:
      1. Exact prompt match
      2. First substring match (key found anywhere in the prompt)
      3. Default response

    Structured calls return a dict; the base class will validate it
    against the requested Pydantic schema, so tests still exercise
    the validation path. Records every call in `self.calls` for
    assertion in tests.
    """

    def __init__(
        self,
        text_responses: dict[str, str] | None = None,
        structured_responses: dict[str, dict] | None = None,
        default_text: str = "ok",
        default_structured: dict | None = None,
    ) -> None:
        super().__init__(model="fake", cache=None)
        self.text_responses = text_responses or {}
        self.structured_responses = structured_responses or {}
        self.default_text = default_text
        self.default_structured = default_structured or {}
        self.calls: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @staticmethod
    def _match(prompt: str, table: dict[str, Any]) -> Any | None:
        if prompt in table:
            return table[prompt]
        for key, value in table.items():
            if key and key in prompt:
                return value
        return None

    def _raw_complete(
        self, prompt: str, system: str | None, temperature: float
    ) -> str:
        self.calls.append(
            {"kind": "text", "prompt": prompt, "system": system}
        )
        match = self._match(prompt, self.text_responses)
        return match if match is not None else self.default_text

    def _raw_structured(
        self,
        prompt: str,
        system: str | None,
        schema: dict,
        temperature: float,
        strict: bool = True,
    ) -> dict:
        self.calls.append(
            {
                "kind": "structured",
                "prompt": prompt,
                "system": system,
                "strict": strict,
            }
        )
        match = self._match(prompt, self.structured_responses)
        return match if match is not None else self.default_structured


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def make_client(
    provider: str | None = None,
    model: str | None = None,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
) -> LLMClient:
    """Construct an LLMClient based on env vars and arguments.

    Provider resolution order:
      1. Explicit `provider` argument
      2. CONVGEN_LLM_PROVIDER env var
      3. Whichever API key is set (Anthropic preferred)
      4. Raise LLMError
    """
    cache = DiskCache(cache_dir) if cache_dir else None
    if provider is None:
        provider = os.environ.get("CONVGEN_LLM_PROVIDER")
    if provider is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise LLMError(
                "No LLM provider configured. Set ANTHROPIC_API_KEY or "
                "OPENAI_API_KEY in .env, or pass provider= explicitly."
            )

    if provider == "anthropic":
        return AnthropicClient(
            model=model or "claude-haiku-4-5-20251001", cache=cache
        )
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini", cache=cache)
    raise LLMError(f"Unknown provider: {provider}")
