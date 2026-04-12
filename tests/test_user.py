"""Tests for the user agent. FakeLLMClient throughout — no network."""

from __future__ import annotations

import pytest

from convgen.agents.planner import ConversationPlan
from convgen.agents.user import (
    USER_SYSTEM_PROMPT,
    UserAgent,
    _build_clarification_response_prompt,
    _build_initial_prompt,
)
from convgen.llm import FakeLLMClient


# ---------------------- fixture helpers ----------------------


def _plan(**overrides) -> ConversationPlan:
    base = dict(
        user_intent="I want to book a hotel in Paris for next weekend",
        persona="A mid-thirties consultant on a short personal trip",
        needs_clarification=False,
        clarification_question="",
        withheld_parameters=[],
    )
    base.update(overrides)
    return ConversationPlan(**base)


# ---------------------- initial prompt building ----------------------


class TestBuildInitialPrompt:
    def test_includes_persona_and_intent(self):
        prompt = _build_initial_prompt(_plan())
        assert "consultant" in prompt
        assert "Paris" in prompt

    def test_no_clarification_instructs_specificity(self):
        prompt = _build_initial_prompt(_plan())
        assert "concrete information" in prompt
        assert "deliberately do NOT mention" not in prompt

    def test_clarification_instructs_withholding(self):
        prompt = _build_initial_prompt(
            _plan(
                needs_clarification=True,
                clarification_question="What's your budget?",
                withheld_parameters=["max_price", "check_in"],
            )
        )
        assert "deliberately do NOT mention" in prompt
        assert "max_price" in prompt
        assert "check_in" in prompt

    def test_clarification_without_params_falls_through_to_specific(self):
        prompt = _build_initial_prompt(
            _plan(
                needs_clarification=True,
                clarification_question="budget?",
                withheld_parameters=[],
            )
        )
        assert "concrete information" in prompt


# ---------------------- clarification-response prompt building ----------------------


class TestBuildClarificationResponsePrompt:
    def test_extracts_most_recent_clarification_question(self):
        plan = _plan(
            needs_clarification=True,
            clarification_question="What's your budget?",
            withheld_parameters=["max_price"],
        )
        history = [
            {"role": "user", "content": "Find me a hotel in Paris"},
            {
                "role": "assistant",
                "action": "clarify",
                "content": "What's your budget range per night?",
            },
        ]
        prompt = _build_clarification_response_prompt(plan, history)
        assert "What's your budget range per night?" in prompt
        assert "max_price" in prompt

    def test_skips_non_clarify_assistant_turns(self):
        plan = _plan(needs_clarification=True, withheld_parameters=["city"])
        history = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "action": "tool_call",
                "endpoint": "x.y",
                "arguments": {},
            },
            {
                "role": "assistant",
                "action": "clarify",
                "content": "Which city?",
            },
        ]
        prompt = _build_clarification_response_prompt(plan, history)
        assert "Which city?" in prompt
        assert "tool_call" not in prompt

    def test_handles_empty_history_gracefully(self):
        plan = _plan(needs_clarification=True, withheld_parameters=["city"])
        prompt = _build_clarification_response_prompt(plan, [])
        assert "The assistant just asked" in prompt


# ---------------------- UserAgent — initial_message ----------------------


class TestUserAgentInitialMessage:
    def test_returns_stripped_string(self):
        llm = FakeLLMClient(default_text="  Find me a hotel in Paris.  ")
        agent = UserAgent(llm)
        out = agent.initial_message(_plan())
        assert out == "Find me a hotel in Paris."

    def test_uses_user_system_prompt(self):
        llm = FakeLLMClient(default_text="hi")
        agent = UserAgent(llm)
        agent.initial_message(_plan())
        assert llm.calls[0]["system"] == USER_SYSTEM_PROMPT

    def test_uses_free_text_not_structured(self):
        llm = FakeLLMClient(default_text="hi")
        agent = UserAgent(llm)
        agent.initial_message(_plan())
        assert llm.calls[0]["kind"] == "text"

    def test_withholding_prompt_sent_to_llm(self):
        llm = FakeLLMClient(default_text="...")
        agent = UserAgent(llm)
        agent.initial_message(
            _plan(
                needs_clarification=True,
                withheld_parameters=["max_price"],
            )
        )
        sent_prompt = llm.calls[0]["prompt"]
        assert "do NOT mention" in sent_prompt
        assert "max_price" in sent_prompt


# ---------------------- UserAgent — clarification_response ----------------------


class TestUserAgentClarificationResponse:
    def test_returns_stripped_string(self):
        llm = FakeLLMClient(default_text="  Under 200 a night.  ")
        agent = UserAgent(llm)
        out = agent.clarification_response(
            _plan(needs_clarification=True, withheld_parameters=["max_price"]),
            history=[
                {"role": "user", "content": "Find me a hotel in Paris"},
                {
                    "role": "assistant",
                    "action": "clarify",
                    "content": "What's your budget?",
                },
            ],
        )
        assert out == "Under 200 a night."

    def test_raises_if_plan_has_no_clarification(self):
        llm = FakeLLMClient(default_text="x")
        agent = UserAgent(llm)
        with pytest.raises(ValueError, match="needs_clarification"):
            agent.clarification_response(_plan(), history=[])

    def test_prompt_contains_history_question(self):
        llm = FakeLLMClient(default_text="x")
        agent = UserAgent(llm)
        agent.clarification_response(
            _plan(needs_clarification=True, withheld_parameters=["city"]),
            history=[
                {
                    "role": "assistant",
                    "action": "clarify",
                    "content": "Which city should I search in?",
                }
            ],
        )
        sent = llm.calls[0]["prompt"]
        assert "Which city should I search in?" in sent
        assert "city" in sent


# ---------------------- temperature ----------------------


class _CapturingFake(FakeLLMClient):
    def _raw_complete(self, prompt, system, temperature):
        self.calls.append(
            {
                "kind": "text",
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
            }
        )
        return self.default_text


class TestUserAgentTemperature:
    def test_default_temperature_is_nonzero(self):
        llm = _CapturingFake(default_text="hi")
        agent = UserAgent(llm)
        agent.initial_message(_plan())
        assert llm.calls[0]["temperature"] > 0

    def test_custom_temperature_respected(self):
        llm = _CapturingFake(default_text="hi")
        agent = UserAgent(llm, temperature=0.3)
        agent.initial_message(_plan())
        assert llm.calls[0]["temperature"] == 0.3
