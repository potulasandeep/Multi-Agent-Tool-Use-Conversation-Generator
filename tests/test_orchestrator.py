"""Tests for the orchestrator. FakeLLMClient drives all agents."""

from __future__ import annotations

import pytest

from convgen.agents.assistant import AssistantAgent
from convgen.agents.planner import PlannerAgent
from convgen.agents.user import UserAgent
from convgen.executor.mock import ExecutorError, MockExecutor
from convgen.graph.sampler import SampledChain
from convgen.llm import FakeLLMClient
from convgen.orchestrator import Conversation, Orchestrator, _internal_to_output
from convgen.registry.models import Endpoint, Parameter, ResponseField


def _ep(
    tool: str,
    name: str,
    required: list[str] | None = None,
    response: list[str] | None = None,
) -> Endpoint:
    return Endpoint(
        id=f"{tool}.{name}",
        tool_name=tool,
        name=name,
        category="Travel",
        parameters=[Parameter(name=n, type="string", required=True) for n in (required or [])],
        response_fields=[ResponseField(name=n, type="string") for n in (response or [])],
    )


def _chain(*endpoints: Endpoint) -> SampledChain:
    return SampledChain(endpoints=list(endpoints), pattern="sequential")


def _plan_dict(**overrides) -> dict:
    base = {
        "user_intent": "I want to book a hotel in Paris for next weekend",
        "persona": "A mid-thirties consultant on a short trip",
        "needs_clarification": False,
        "clarification_question": "",
        "withheld_parameters": [],
    }
    base.update(overrides)
    return base


def _clarify_action() -> dict:
    return {
        "action": "clarify",
        "clarification": "What's your budget range per night?",
        "tool_endpoint": "",
        "tool_arguments": {},
        "final_answer": "",
    }


def _tool_call_action(arguments: dict) -> dict:
    return {
        "action": "tool_call",
        "clarification": "",
        "tool_endpoint": "",
        "tool_arguments": arguments,
        "final_answer": "",
    }


def _final_action(text: str = "All done! Your booking is confirmed.") -> dict:
    return {
        "action": "final_answer",
        "clarification": "",
        "tool_endpoint": "",
        "tool_arguments": {},
        "final_answer": text,
    }


def _fake_llm(
    plan_dict: dict | None = None,
    clarify_dict: dict | None = None,
    tool_call_by_endpoint: dict[str, dict] | None = None,
    final_dict: dict | None = None,
    user_initial_text: str = "Find me a hotel in Paris.",
    user_clarification_text: str = "Paris, and under 200 a night.",
) -> FakeLLMClient:
    structured: dict[str, dict] = {
        "Design a realistic user scenario": plan_dict or _plan_dict(),
        "action=clarify and ask a natural": clarify_dict or _clarify_action(),
        "action=final_answer and write": final_dict or _final_action(),
    }

    if tool_call_by_endpoint:
        for endpoint_id, action in tool_call_by_endpoint.items():
            structured[f"`{endpoint_id}`"] = action

    return FakeLLMClient(
        structured_responses=structured,
        text_responses={
            "OPENING message": user_initial_text,
            "assistant just asked": user_clarification_text,
        },
        default_structured=_final_action(),
        default_text="...",
    )


def _build_orchestrator(llm: FakeLLMClient, clarification_rate: float = 0.3) -> Orchestrator:
    return Orchestrator(
        planner=PlannerAgent(llm),
        user=UserAgent(llm),
        assistant=AssistantAgent(llm),
        executor=MockExecutor(seed=0),
        clarification_rate=clarification_rate,
    )


class TestInternalToOutput:
    def test_user_turn_passthrough(self):
        out = _internal_to_output([{"role": "user", "content": "hi"}])
        assert out == [{"role": "user", "content": "hi"}]

    def test_assistant_clarify_drops_action_tag(self):
        out = _internal_to_output([{"role": "assistant", "action": "clarify", "content": "?"}])
        assert out == [{"role": "assistant", "content": "?"}]

    def test_assistant_tool_call_uses_spec_shape(self):
        out = _internal_to_output(
            [
                {
                    "role": "assistant",
                    "action": "tool_call",
                    "endpoint": "hotels.book",
                    "arguments": {"hotel_id": "x"},
                }
            ]
        )
        assert out == [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"endpoint": "hotels.book", "arguments": {"hotel_id": "x"}}],
            }
        ]

    def test_assistant_final_answer_is_plain_assistant(self):
        out = _internal_to_output(
            [{"role": "assistant", "action": "final_answer", "content": "done"}]
        )
        assert out == [{"role": "assistant", "content": "done"}]

    def test_tool_role_drops_endpoint_from_output(self):
        out = _internal_to_output([{"role": "tool", "endpoint": "x.y", "content": {"results": []}}])
        assert out == [{"role": "tool", "content": {"results": []}}]


class TestHappyPathNoClarification:
    def test_produces_conversation(self):
        chain = _chain(
            _ep("hotels", "search", required=["city"], response=["id", "name"]),
            _ep("hotels", "book", required=["hotel_id", "check_in"]),
        )
        llm = _fake_llm(
            plan_dict=_plan_dict(needs_clarification=False),
            tool_call_by_endpoint={
                "hotels.search": _tool_call_action({"city": "Paris"}),
                "hotels.book": _tool_call_action({"hotel_id": "hot_1234", "check_in": "2026-04-11"}),
            },
        )
        orch = _build_orchestrator(llm, clarification_rate=0.0)
        convo = orch.run(chain, seed=42, conversation_id="conv_test_1")

        assert isinstance(convo, Conversation)
        assert convo.conversation_id == "conv_test_1"
        assert convo.judge_scores is None
        assert convo.metadata["failed"] is False
        assert convo.metadata["had_clarification"] is False

    def test_message_sequence(self):
        chain = _chain(
            _ep("hotels", "search", required=["city"], response=["id"]),
            _ep("hotels", "book", required=["hotel_id"]),
        )
        llm = _fake_llm(
            plan_dict=_plan_dict(needs_clarification=False),
            tool_call_by_endpoint={
                "hotels.search": _tool_call_action({"city": "Paris"}),
                "hotels.book": _tool_call_action({"hotel_id": "hot_1234"}),
            },
        )
        orch = _build_orchestrator(llm, clarification_rate=0.0)
        convo = orch.run(chain, seed=1, conversation_id="c1")

        roles = [m["role"] for m in convo.messages]
        assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
        assert convo.messages[1]["content"] is None
        assert "tool_calls" in convo.messages[1]
        assert convo.messages[3]["content"] is None
        assert convo.messages[-1]["content"] is not None
        assert isinstance(convo.messages[-1]["content"], str)


class TestClarificationPath:
    def test_adds_clarify_pair_when_plan_requires(self):
        chain = _chain(_ep("hotels", "search", required=["city"], response=["id"]))
        llm = _fake_llm(
            plan_dict=_plan_dict(
                needs_clarification=True,
                clarification_question="What's your budget?",
                withheld_parameters=["max_price"],
            ),
            tool_call_by_endpoint={"hotels.search": _tool_call_action({"city": "Paris"})},
        )
        orch = _build_orchestrator(llm, clarification_rate=1.0)
        convo = orch.run(chain, seed=1, conversation_id="c1")

        roles = [m["role"] for m in convo.messages]
        assert roles == ["user", "assistant", "user", "assistant", "tool", "assistant"]
        assert convo.messages[1]["content"] == "What's your budget range per night?"
        assert convo.metadata["had_clarification"] is True


class TestReproducibility:
    def test_same_seed_same_clarification_decision(self):
        chain = _chain(_ep("hotels", "search", required=["city"], response=["id"]))
        llm = _fake_llm(
            plan_dict=_plan_dict(
                needs_clarification=True,
                clarification_question="?",
                withheld_parameters=["city"],
            ),
            tool_call_by_endpoint={"hotels.search": _tool_call_action({"city": "Paris"})},
        )
        orch = _build_orchestrator(llm, clarification_rate=0.5)
        a = orch.run(chain, seed=42, conversation_id="a")
        b = orch.run(chain, seed=42, conversation_id="b")
        assert a.metadata["force_clarification"] == b.metadata["force_clarification"]

    def test_clarification_rate_roughly_targeted(self):
        chain = _chain(_ep("hotels", "search", required=["city"], response=["id"]))
        llm = _fake_llm(
            plan_dict=_plan_dict(
                needs_clarification=True,
                clarification_question="?",
                withheld_parameters=["city"],
            ),
            tool_call_by_endpoint={"hotels.search": _tool_call_action({"city": "Paris"})},
        )
        orch = _build_orchestrator(llm, clarification_rate=0.3)

        forced = 0
        trials = 200
        for i in range(trials):
            convo = orch.run(chain, seed=i, conversation_id=f"c{i}")
            if convo.metadata["force_clarification"]:
                forced += 1
        rate = forced / trials
        assert 0.2 <= rate <= 0.4


class _FailingExecutor:
    def __init__(self, fail_on_index: int) -> None:
        self._real = MockExecutor(seed=0)
        self._fail_on = fail_on_index
        self._calls = 0

    def execute(self, endpoint, arguments, session):
        idx = self._calls
        self._calls += 1
        if idx == self._fail_on:
            raise ExecutorError(f"simulated failure on call #{idx} ({endpoint.id})")
        return self._real.execute(endpoint, arguments, session)


class TestExecutorFailure:
    def test_failure_records_reason_and_continues_to_final(self):
        chain = _chain(
            _ep("hotels", "search", required=["city"], response=["id"]),
            _ep("hotels", "book", required=["hotel_id", "check_in"]),
        )
        llm = _fake_llm(
            plan_dict=_plan_dict(needs_clarification=False),
            tool_call_by_endpoint={
                "hotels.search": _tool_call_action({"city": "Paris"}),
                "hotels.book": _tool_call_action({"hotel_id": "x", "check_in": "y"}),
            },
        )
        orch = Orchestrator(
            planner=PlannerAgent(llm),
            user=UserAgent(llm),
            assistant=AssistantAgent(llm),
            executor=_FailingExecutor(fail_on_index=1),
            clarification_rate=0.0,
        )
        convo = orch.run(chain, seed=1, conversation_id="c_fail")

        assert convo.metadata["failed"] is True
        assert "simulated failure" in convo.metadata["failure_reason"]
        assert convo.messages[-1]["role"] == "assistant"
        assert convo.messages[-1]["content"] is not None

        tool_msgs = [m for m in convo.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert "error" in tool_msgs[1]["content"]

    def test_failure_on_first_call_still_produces_final_answer(self):
        chain = _chain(
            _ep("hotels", "search", required=["city"], response=["id"]),
            _ep("hotels", "book", required=["hotel_id"]),
        )
        llm = _fake_llm(
            plan_dict=_plan_dict(needs_clarification=False),
            tool_call_by_endpoint={
                "hotels.search": _tool_call_action({"city": "Paris"}),
                "hotels.book": _tool_call_action({"hotel_id": "x"}),
            },
        )
        orch = Orchestrator(
            planner=PlannerAgent(llm),
            user=UserAgent(llm),
            assistant=AssistantAgent(llm),
            executor=_FailingExecutor(fail_on_index=0),
            clarification_rate=0.0,
        )
        convo = orch.run(chain, seed=1, conversation_id="c_fail_0")
        assert convo.metadata["failed"] is True
        assert convo.messages[-1]["role"] == "assistant"
        assert len(convo.metadata["endpoints_called"]) == 1


class TestMetadata:
    def test_metadata_has_all_expected_fields(self):
        chain = _chain(
            _ep("hotels", "search", required=["city"], response=["id"]),
            _ep("hotels", "book", required=["hotel_id"]),
        )
        llm = _fake_llm(
            plan_dict=_plan_dict(needs_clarification=False),
            tool_call_by_endpoint={
                "hotels.search": _tool_call_action({"city": "Paris"}),
                "hotels.book": _tool_call_action({"hotel_id": "hot_1"}),
            },
        )
        orch = _build_orchestrator(llm, clarification_rate=0.0)
        convo = orch.run(chain, seed=7, conversation_id="c_meta")

        md = convo.metadata
        expected_keys = {
            "seed",
            "chain_endpoint_ids",
            "chain_length",
            "pattern",
            "tools_used",
            "endpoints_called",
            "num_turns",
            "had_clarification",
            "force_clarification",
            "failed",
            "failure_reason",
            "persona",
            "user_intent",
        }
        assert expected_keys.issubset(md.keys())
        assert md["seed"] == 7
        assert md["chain_length"] == 2
        assert md["pattern"] == "sequential"
        assert md["tools_used"] == ["hotels"]
        assert md["endpoints_called"] == ["hotels.search", "hotels.book"]
        assert md["num_turns"] == len(convo.messages)
        assert md["persona"]
        assert md["user_intent"]

    def test_tools_used_deduplicates(self):
        chain = _chain(
            _ep("hotels", "search", required=["city"], response=["id"]),
            _ep("hotels", "get_details", required=["hotel_id"]),
            _ep("hotels", "book", required=["hotel_id"]),
        )
        llm = _fake_llm(
            plan_dict=_plan_dict(needs_clarification=False),
            tool_call_by_endpoint={
                "hotels.search": _tool_call_action({"city": "Paris"}),
                "hotels.get_details": _tool_call_action({"hotel_id": "hot_1"}),
                "hotels.book": _tool_call_action({"hotel_id": "hot_1"}),
            },
        )
        orch = _build_orchestrator(llm, clarification_rate=0.0)
        convo = orch.run(chain, seed=1, conversation_id="c_dedup")
        assert convo.metadata["tools_used"] == ["hotels"]


class TestEdgeCases:
    def test_empty_chain_rejected(self):
        llm = _fake_llm()
        orch = _build_orchestrator(llm)
        empty = SampledChain(endpoints=[], pattern="sequential")
        with pytest.raises(ValueError, match="empty chain"):
            orch.run(empty, seed=0, conversation_id="c_empty")

    def test_invalid_clarification_rate_rejected(self):
        llm = _fake_llm()
        with pytest.raises(ValueError, match="clarification_rate"):
            Orchestrator(
                planner=PlannerAgent(llm),
                user=UserAgent(llm),
                assistant=AssistantAgent(llm),
                executor=MockExecutor(seed=0),
                clarification_rate=1.5,
            )

    def test_single_endpoint_chain(self):
        chain = _chain(_ep("weather", "forecast", required=["city"], response=["temp"]))
        llm = _fake_llm(
            plan_dict=_plan_dict(
                user_intent="What's the weather like in Paris tomorrow?",
                persona="A traveler checking conditions",
                needs_clarification=False,
            ),
            tool_call_by_endpoint={"weather.forecast": _tool_call_action({"city": "Paris"})},
        )
        orch = _build_orchestrator(llm, clarification_rate=0.0)
        convo = orch.run(chain, seed=1, conversation_id="c_single")
        assert convo.metadata["chain_length"] == 1
        assert convo.metadata["failed"] is False
