"""Tests for the Assistant agent. FakeLLMClient throughout — no network."""

from __future__ import annotations

import logging

import pytest

from convgen.agents.assistant import (
    ASSISTANT_SYSTEM_PROMPT,
    AssistantAction,
    AssistantAgent,
    ExpectedMode,
    _build_assistant_prompt,
    _format_history,
    _format_session_values,
    _normalize_action,
)
from convgen.agents.planner import ConversationPlan
from convgen.executor.mock import SessionStore
from convgen.llm import FakeLLMClient
from convgen.registry.models import Endpoint, Parameter


# ---------------------- fixture helpers ----------------------


def _plan(**overrides) -> ConversationPlan:
    base = dict(
        user_intent="I want to book a hotel in Paris for next weekend",
        persona="A mid-thirties consultant on a short trip",
        needs_clarification=False,
        clarification_question="",
        withheld_parameters=[],
    )
    base.update(overrides)
    return ConversationPlan(**base)


def _ep(
    tool: str = "hotels",
    name: str = "book",
    required: list[str] | None = None,
    optional: list[str] | None = None,
) -> Endpoint:
    return Endpoint(
        id=f"{tool}.{name}",
        tool_name=tool,
        name=name,
        category="Travel",
        parameters=[
            Parameter(name=n, type="string", required=True)
            for n in (required or [])
        ]
        + [
            Parameter(name=n, type="string", required=False)
            for n in (optional or [])
        ],
    )


def _populated_session() -> SessionStore:
    s = SessionStore()
    s.record("id", "hot_2824", tool_name="hotels")
    s.record("id", "hot_5506", tool_name="hotels")
    s.record("name", "Allison Hill", tool_name="hotels")
    s.record("name", "Noah Rhodes", tool_name="hotels")
    return s


# ---------------------- history formatting ----------------------


class TestFormatHistory:
    def test_empty_history(self):
        assert "no messages yet" in _format_history([])

    def test_user_turn(self):
        out = _format_history([{"role": "user", "content": "hi"}])
        assert "User: hi" in out

    def test_assistant_clarify(self):
        out = _format_history(
            [{"role": "assistant", "action": "clarify", "content": "budget?"}]
        )
        assert "[clarify]" in out
        assert "budget?" in out

    def test_assistant_tool_call(self):
        out = _format_history(
            [
                {
                    "role": "assistant",
                    "action": "tool_call",
                    "endpoint": "hotels.search",
                    "arguments": {"city": "Paris"},
                }
            ]
        )
        assert "[tool_call hotels.search]" in out
        assert "Paris" in out

    def test_tool_output_truncated_when_long(self):
        long_content = {"results": ["x" * 1000]}
        out = _format_history(
            [{"role": "tool", "endpoint": "x.y", "content": long_content}]
        )
        assert "truncated" in out
        assert len(out) < 2000


# ---------------------- session-value formatting ----------------------


class TestFormatSessionValues:
    def test_empty_session(self):
        assert "no values" in _format_session_values(SessionStore())

    def test_keys_present(self):
        out = _format_session_values(_populated_session())
        assert "hotels.id" in out
        assert "hot_2824" in out
        assert "Allison Hill" in out

    def test_keys_are_sorted(self):
        out = _format_session_values(_populated_session())
        lines = [line for line in out.splitlines() if ":" in line]
        keys = [line.split(":", 1)[0].strip() for line in lines]
        assert keys == sorted(keys)


# ---------------------- prompt building ----------------------


class TestBuildPrompt:
    def test_clarify_prompt_contains_suggested_question(self):
        prompt = _build_assistant_prompt(
            plan=_plan(
                needs_clarification=True,
                clarification_question="What's your budget?",
            ),
            history=[{"role": "user", "content": "Find me a hotel"}],
            session=SessionStore(),
            mode=ExpectedMode.CLARIFY,
            next_endpoint=None,
        )
        assert "action=clarify" in prompt
        assert "What's your budget?" in prompt

    def test_tool_call_prompt_lists_required_params(self):
        ep = _ep(required=["hotel_id", "check_in"])
        prompt = _build_assistant_prompt(
            plan=_plan(),
            history=[],
            session=_populated_session(),
            mode=ExpectedMode.TOOL_CALL,
            next_endpoint=ep,
        )
        assert "hotels.book" in prompt
        assert "hotel_id" in prompt
        assert "check_in" in prompt

    def test_tool_call_prompt_includes_session_values(self):
        prompt = _build_assistant_prompt(
            plan=_plan(),
            history=[],
            session=_populated_session(),
            mode=ExpectedMode.TOOL_CALL,
            next_endpoint=_ep(required=["hotel_id"]),
        )
        assert "hot_2824" in prompt
        assert "hotels.id" in prompt

    def test_final_answer_prompt(self):
        prompt = _build_assistant_prompt(
            plan=_plan(),
            history=[],
            session=_populated_session(),
            mode=ExpectedMode.FINAL_ANSWER,
            next_endpoint=None,
        )
        assert "action=final_answer" in prompt
        assert "wrap-up" in prompt

    def test_tool_call_without_next_endpoint_raises(self):
        with pytest.raises(ValueError, match="requires a next_endpoint"):
            _build_assistant_prompt(
                plan=_plan(),
                history=[],
                session=SessionStore(),
                mode=ExpectedMode.TOOL_CALL,
                next_endpoint=None,
            )


# ---------------------- normalize_action ----------------------


class TestNormalizeAction:
    def test_clarify_zeroes_other_fields(self):
        action = AssistantAction(
            action=ExpectedMode.CLARIFY,
            clarification="What's your budget?",
            tool_endpoint="garbage",
            tool_arguments={"x": 1},
            final_answer="spurious",
        )
        out = _normalize_action(action, ExpectedMode.CLARIFY, None)
        assert out.clarification == "What's your budget?"
        assert out.tool_endpoint == ""
        assert out.tool_arguments == {}
        assert out.final_answer == ""

    def test_clarify_with_empty_clarification_gets_fallback(self):
        action = AssistantAction(action=ExpectedMode.CLARIFY)
        out = _normalize_action(action, ExpectedMode.CLARIFY, None)
        assert out.clarification

    def test_tool_call_forces_endpoint_to_next(self):
        ep = _ep(required=["hotel_id"])
        action = AssistantAction(
            action=ExpectedMode.TOOL_CALL,
            tool_endpoint="wrong.endpoint",
            tool_arguments={"hotel_id": "hot_2824"},
        )
        out = _normalize_action(action, ExpectedMode.TOOL_CALL, ep)
        assert out.tool_endpoint == ep.id
        assert out.tool_arguments == {"hotel_id": "hot_2824"}

    def test_tool_call_preserves_llm_arguments(self):
        ep = _ep(required=["hotel_id"])
        action = AssistantAction(
            action=ExpectedMode.TOOL_CALL,
            tool_endpoint=ep.id,
            tool_arguments={"hotel_id": "hot_9999", "check_in": "2026-04-11"},
        )
        out = _normalize_action(action, ExpectedMode.TOOL_CALL, ep)
        assert out.tool_arguments["hotel_id"] == "hot_9999"
        assert out.tool_arguments["check_in"] == "2026-04-11"

    def test_tool_call_zeroes_non_tool_fields(self):
        ep = _ep(required=["hotel_id"])
        action = AssistantAction(
            action=ExpectedMode.TOOL_CALL,
            tool_endpoint=ep.id,
            tool_arguments={"hotel_id": "hot_2824"},
            clarification="spurious",
            final_answer="spurious",
        )
        out = _normalize_action(action, ExpectedMode.TOOL_CALL, ep)
        assert out.clarification == ""
        assert out.final_answer == ""

    def test_final_answer_zeroes_other_fields(self):
        action = AssistantAction(
            action=ExpectedMode.FINAL_ANSWER,
            final_answer="All booked! Confirmation hot_8472.",
            tool_endpoint="spurious",
            tool_arguments={"x": 1},
            clarification="spurious",
        )
        out = _normalize_action(action, ExpectedMode.FINAL_ANSWER, None)
        assert out.final_answer == "All booked! Confirmation hot_8472."
        assert out.tool_endpoint == ""
        assert out.tool_arguments == {}
        assert out.clarification == ""

    def test_mode_mismatch_is_overridden_with_warning(self, caplog):
        action = AssistantAction(
            action=ExpectedMode.CLARIFY,
            clarification="wrong turn",
        )
        with caplog.at_level(logging.WARNING):
            out = _normalize_action(action, ExpectedMode.FINAL_ANSWER, None)
        assert out.action == ExpectedMode.FINAL_ANSWER
        assert any("overriding" in r.message for r in caplog.records)

    def test_endpoint_mismatch_logged(self, caplog):
        ep = _ep(required=["hotel_id"])
        action = AssistantAction(
            action=ExpectedMode.TOOL_CALL,
            tool_endpoint="something.else",
            tool_arguments={"hotel_id": "x"},
        )
        with caplog.at_level(logging.WARNING):
            out = _normalize_action(action, ExpectedMode.TOOL_CALL, ep)
        assert out.tool_endpoint == ep.id
        assert any("chose endpoint" in r.message for r in caplog.records)


# ---------------------- AssistantAgent integration ----------------------


def _action_dict(**kwargs) -> dict:
    base = {
        "action": "clarify",
        "clarification": "",
        "tool_endpoint": "",
        "tool_arguments": {},
        "final_answer": "",
    }
    base.update(kwargs)
    return base


class TestAssistantAgentClarify:
    def test_happy_path(self):
        llm = FakeLLMClient(
            default_structured=_action_dict(
                action="clarify",
                clarification="What's your budget range per night?",
            )
        )
        agent = AssistantAgent(llm)
        action = agent.step(
            plan=_plan(
                needs_clarification=True,
                clarification_question="What's your budget?",
            ),
            history=[{"role": "user", "content": "Find me a hotel in Paris"}],
            session=SessionStore(),
            mode=ExpectedMode.CLARIFY,
        )
        assert action.action == ExpectedMode.CLARIFY
        assert "budget" in action.clarification.lower()

    def test_system_prompt_used(self):
        llm = FakeLLMClient(
            default_structured=_action_dict(action="clarify", clarification="?")
        )
        agent = AssistantAgent(llm)
        agent.step(
            plan=_plan(),
            history=[],
            session=SessionStore(),
            mode=ExpectedMode.CLARIFY,
        )
        assert llm.calls[0]["system"] == ASSISTANT_SYSTEM_PROMPT


class TestAssistantAgentToolCall:
    def test_happy_path(self):
        llm = FakeLLMClient(
            default_structured=_action_dict(
                action="tool_call",
                tool_endpoint="hotels.book",
                tool_arguments={"hotel_id": "hot_2824", "check_in": "2026-04-11"},
            )
        )
        agent = AssistantAgent(llm)
        ep = _ep(required=["hotel_id", "check_in"])
        action = agent.step(
            plan=_plan(),
            history=[{"role": "user", "content": "Book the first one"}],
            session=_populated_session(),
            mode=ExpectedMode.TOOL_CALL,
            next_endpoint=ep,
        )
        assert action.action == ExpectedMode.TOOL_CALL
        assert action.tool_endpoint == "hotels.book"
        assert action.tool_arguments["hotel_id"] == "hot_2824"

    def test_grounding_block_is_in_prompt(self):
        llm = FakeLLMClient(
            default_structured=_action_dict(
                action="tool_call",
                tool_endpoint="hotels.book",
                tool_arguments={"hotel_id": "hot_2824"},
            )
        )
        agent = AssistantAgent(llm)
        agent.step(
            plan=_plan(),
            history=[],
            session=_populated_session(),
            mode=ExpectedMode.TOOL_CALL,
            next_endpoint=_ep(required=["hotel_id"]),
        )
        prompt = llm.calls[0]["prompt"]
        assert "Available values from previous tool calls" in prompt
        assert "hot_2824" in prompt

    def test_missing_next_endpoint_raises(self):
        llm = FakeLLMClient(default_structured=_action_dict())
        agent = AssistantAgent(llm)
        with pytest.raises(ValueError, match="requires next_endpoint"):
            agent.step(
                plan=_plan(),
                history=[],
                session=SessionStore(),
                mode=ExpectedMode.TOOL_CALL,
                next_endpoint=None,
            )

    def test_hallucinated_endpoint_overridden(self):
        llm = FakeLLMClient(
            default_structured=_action_dict(
                action="tool_call",
                tool_endpoint="garbage.nope",
                tool_arguments={"hotel_id": "hot_2824"},
            )
        )
        agent = AssistantAgent(llm)
        ep = _ep(required=["hotel_id"])
        action = agent.step(
            plan=_plan(),
            history=[],
            session=_populated_session(),
            mode=ExpectedMode.TOOL_CALL,
            next_endpoint=ep,
        )
        assert action.tool_endpoint == ep.id


class TestAssistantAgentFinalAnswer:
    def test_happy_path(self):
        llm = FakeLLMClient(
            default_structured=_action_dict(
                action="final_answer",
                final_answer="Booked! Your confirmation is hot_8472.",
            )
        )
        agent = AssistantAgent(llm)
        action = agent.step(
            plan=_plan(),
            history=[{"role": "user", "content": "please book it"}],
            session=_populated_session(),
            mode=ExpectedMode.FINAL_ANSWER,
        )
        assert action.action == ExpectedMode.FINAL_ANSWER
        assert "hot_8472" in action.final_answer

    def test_empty_final_answer_gets_fallback(self):
        llm = FakeLLMClient(
            default_structured=_action_dict(
                action="final_answer",
                final_answer="",
            )
        )
        agent = AssistantAgent(llm)
        action = agent.step(
            plan=_plan(),
            history=[],
            session=SessionStore(),
            mode=ExpectedMode.FINAL_ANSWER,
        )
        assert action.final_answer


class TestAssistantAgentCrossMode:
    def test_llm_returns_wrong_mode_but_normalizer_fixes_it(self, caplog):
        llm = FakeLLMClient(
            default_structured=_action_dict(
                action="clarify",
                clarification="I'm confused",
            )
        )
        agent = AssistantAgent(llm)
        with caplog.at_level(logging.WARNING):
            action = agent.step(
                plan=_plan(),
                history=[],
                session=_populated_session(),
                mode=ExpectedMode.FINAL_ANSWER,
            )
        assert action.action == ExpectedMode.FINAL_ANSWER
        assert any("overriding" in r.message for r in caplog.records)
