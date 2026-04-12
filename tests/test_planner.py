"""Tests for the planner agent. No real LLM calls — FakeLLMClient only."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from convgen.agents.planner import (
    ConversationPlan,
    PLANNER_SYSTEM_PROMPT,
    PlannerAgent,
    _build_planner_prompt,
    _format_chain_for_prompt,
)
from convgen.graph.sampler import SampledChain
from convgen.llm import FakeLLMClient
from convgen.registry.models import Endpoint, Parameter


# ---------------------- fixture helpers ----------------------


def _ep(
    tool: str,
    name: str,
    category: str = "Travel",
    required: list[str] | None = None,
    optional: list[str] | None = None,
    description: str = "",
) -> Endpoint:
    return Endpoint(
        id=f"{tool}.{name}",
        tool_name=tool,
        name=name,
        description=description,
        category=category,
        parameters=[
            Parameter(name=name_, type="string", required=True)
            for name_ in (required or [])
        ]
        + [
            Parameter(name=name_, type="string", required=False)
            for name_ in (optional or [])
        ],
    )


def _chain(*endpoints: Endpoint) -> SampledChain:
    return SampledChain(endpoints=list(endpoints), pattern="sequential")


@pytest.fixture
def hotels_chain() -> SampledChain:
    return _chain(
        _ep(
            "hotels",
            "search",
            required=["city"],
            optional=["max_price"],
            description="Search for hotels in a city",
        ),
        _ep("hotels", "get_details", required=["hotel_id"]),
        _ep("hotels", "book", required=["hotel_id", "check_in"]),
    )


def _valid_plan_dict(**overrides) -> dict:
    base = {
        "user_intent": "I need to book a hotel in Paris for next weekend",
        "persona": "A mid-thirties consultant on a short personal trip",
        "needs_clarification": False,
        "clarification_question": "",
        "withheld_parameters": [],
    }
    base.update(overrides)
    return base


# ---------------------- ConversationPlan schema ----------------------


class TestConversationPlanSchema:
    def test_minimal_valid_plan(self):
        plan = ConversationPlan(
            user_intent=(
                "I want to book a hotel in Paris for the upcoming weekend"
            ),
            persona="A weekend tourist planning a short trip",
            needs_clarification=False,
        )
        assert plan.clarification_question == ""
        assert plan.withheld_parameters == []

    def test_too_short_intent_rejected(self):
        with pytest.raises(ValidationError):
            ConversationPlan(
                user_intent="hi",
                persona="a casual user",
                needs_clarification=False,
            )

    def test_too_short_persona_rejected(self):
        with pytest.raises(ValidationError):
            ConversationPlan(
                user_intent=(
                    "I would like to find and book a hotel in Paris"
                ),
                persona="x",
                needs_clarification=False,
            )

    def test_plan_with_clarification_fields(self):
        plan = ConversationPlan(
            user_intent=(
                "Find me a hotel somewhere fun for next weekend"
            ),
            persona="A vague planner with no fixed destination",
            needs_clarification=True,
            clarification_question="Which city are you thinking of?",
            withheld_parameters=["city"],
        )
        assert plan.needs_clarification
        assert "city" in plan.withheld_parameters


# ---------------------- chain formatting ----------------------


class TestFormatChain:
    def test_single_endpoint(self):
        chain = _chain(_ep("hotels", "search", required=["city"]))
        text = _format_chain_for_prompt(chain)
        assert "Step 1:" in text
        assert "hotels.search" in text
        assert "city" in text

    def test_multiple_endpoints_numbered(self, hotels_chain: SampledChain):
        text = _format_chain_for_prompt(hotels_chain)
        assert "Step 1:" in text
        assert "Step 2:" in text
        assert "Step 3:" in text
        assert "hotels.search" in text
        assert "hotels.book" in text

    def test_includes_optional_params_when_present(self):
        chain = _chain(
            _ep(
                "hotels",
                "search",
                required=["city"],
                optional=["max_price"],
            )
        )
        text = _format_chain_for_prompt(chain)
        assert "max_price" in text

    def test_omits_optional_params_when_absent(self):
        chain = _chain(_ep("hotels", "book", required=["hotel_id"]))
        text = _format_chain_for_prompt(chain)
        assert "Optional params" not in text

    def test_includes_category(self):
        chain = _chain(
            _ep("hotels", "search", category="Travel", required=["city"])
        )
        text = _format_chain_for_prompt(chain)
        assert "Travel" in text

    def test_includes_description_when_present(self):
        chain = _chain(
            _ep(
                "hotels",
                "search",
                required=["city"],
                description="Find hotels in a city by name",
            )
        )
        text = _format_chain_for_prompt(chain)
        assert "Find hotels in a city" in text


# ---------------------- prompt building ----------------------


class TestBuildPrompt:
    def test_prompt_includes_chain(self, hotels_chain: SampledChain):
        prompt = _build_planner_prompt(hotels_chain, force_clarification=False)
        assert "hotels.search" in prompt
        assert "hotels.book" in prompt

    def test_force_clarification_branch_explicit(self, hotels_chain: SampledChain):
        with_clarification = _build_planner_prompt(
            hotels_chain, force_clarification=True
        )
        assert "MUST require clarification" in with_clarification
        assert "needs_clarification` to true" in with_clarification

    def test_no_clarification_branch_explicit(self, hotels_chain: SampledChain):
        without = _build_planner_prompt(hotels_chain, force_clarification=False)
        assert "enough information" in without
        assert "needs_clarification` to false" in without
        assert "MUST require clarification" not in without

    def test_prompt_warns_against_tool_name_leakage(
        self, hotels_chain: SampledChain
    ):
        prompt = _build_planner_prompt(
            hotels_chain, force_clarification=False
        )
        assert "Do NOT reference the tool names" in prompt


# ---------------------- PlannerAgent ----------------------


class TestPlannerAgent:
    def test_returns_valid_plan(self, hotels_chain: SampledChain):
        llm = FakeLLMClient(default_structured=_valid_plan_dict())
        agent = PlannerAgent(llm)
        plan = agent.plan(hotels_chain, force_clarification=False)
        assert isinstance(plan, ConversationPlan)
        assert "Paris" in plan.user_intent

    def test_force_clarification_passes_through(self, hotels_chain: SampledChain):
        llm = FakeLLMClient(
            default_structured=_valid_plan_dict(
                user_intent=(
                    "Find me a hotel for next weekend, somewhere nice"
                ),
                persona="A vague weekend planner",
                needs_clarification=True,
                clarification_question="Which city would you like?",
                withheld_parameters=["city"],
            )
        )
        agent = PlannerAgent(llm)
        plan = agent.plan(hotels_chain, force_clarification=True)
        assert plan.needs_clarification is True
        assert plan.clarification_question == "Which city would you like?"
        assert plan.withheld_parameters == ["city"]

    def test_llm_clarification_fields_zeroed_when_not_forced(
        self, hotels_chain: SampledChain
    ):
        # LLM hallucinates clarification fields; planner should normalize.
        llm = FakeLLMClient(
            default_structured=_valid_plan_dict(
                needs_clarification=True,
                clarification_question="spurious question",
                withheld_parameters=["city"],
            )
        )
        agent = PlannerAgent(llm)
        plan = agent.plan(hotels_chain, force_clarification=False)
        assert plan.needs_clarification is False
        assert plan.clarification_question == ""
        assert plan.withheld_parameters == []

    def test_force_clarification_but_llm_refuses_is_overridden(
        self, hotels_chain: SampledChain, caplog: pytest.LogCaptureFixture
    ):
        llm = FakeLLMClient(default_structured=_valid_plan_dict())
        agent = PlannerAgent(llm)
        with caplog.at_level(logging.WARNING):
            plan = agent.plan(hotels_chain, force_clarification=True)
        assert plan.needs_clarification is True
        assert plan.withheld_parameters
        assert any("overriding" in r.message for r in caplog.records)

    def test_llm_is_called_with_system_prompt(self, hotels_chain: SampledChain):
        llm = FakeLLMClient(default_structured=_valid_plan_dict())
        agent = PlannerAgent(llm)
        agent.plan(hotels_chain, force_clarification=False)
        assert len(llm.calls) == 1
        system = llm.calls[0]["system"]
        assert system == PLANNER_SYSTEM_PROMPT

    def test_empty_chain_rejected(self):
        llm = FakeLLMClient(default_structured=_valid_plan_dict())
        agent = PlannerAgent(llm)
        empty_chain = SampledChain(endpoints=[], pattern="sequential")
        with pytest.raises(ValueError, match="empty chain"):
            agent.plan(empty_chain)

    def test_planner_prompt_body_contains_chain_endpoints(
        self, hotels_chain: SampledChain
    ):
        llm = FakeLLMClient(default_structured=_valid_plan_dict())
        agent = PlannerAgent(llm)
        agent.plan(hotels_chain, force_clarification=False)
        call_prompt = llm.calls[0]["prompt"]
        assert "hotels.search" in call_prompt
        assert "hotels.get_details" in call_prompt
        assert "hotels.book" in call_prompt
