"""Planner agent.

Runs once per conversation. Takes a SampledChain and produces a
ConversationPlan describing:

  - what the user wants (in natural language)
  - who the user is (persona)
  - whether the user's first message will withhold required information,
    forcing the assistant to ask a clarifying question
  - what specific parameters are withheld, and what the clarification
    question should be

Design notes
------------
The decision to clarify is NOT made by the LLM. The orchestrator flips a
seeded coin and passes the result in as `force_clarification`. This keeps
the clarification rate controllable and reproducible across runs. The
Planner's only job when clarification is forced is to pick *which*
parameters the user will be vague about and write a natural question for
the assistant to ask.

Output is post-normalized: if the orchestrator said `force_clarification
=False` but the LLM returned clarification fields anyway, we zero them
out. This saves a retry and keeps downstream code simple — it can trust
that `plan.needs_clarification` and `plan.clarification_question`
are always in sync with the orchestrator's decision.

Temperature is 0. Variety across conversations comes from the sampler
picking different chains, not from the Planner being random.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from convgen.graph.sampler import SampledChain
from convgen.llm.client import LLMClient

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------


class ConversationPlan(BaseModel):
    """Blueprint for one conversation, produced once by the Planner.

    Consumed by the User and Assistant agents in later steps. The shape
    is deliberately small: the Planner gives the conversation a spine,
    not a script. The agents fill in the dialogue.
    """

    user_intent: str = Field(
        description=(
            "What the user wants, in plain first-person language. "
            "Must plausibly require the entire tool chain."
        ),
        min_length=10,
    )
    persona: str = Field(
        description=(
            "Short description of the user — who they are, their "
            "context. Grounded and specific, not generic."
        ),
        min_length=5,
    )
    needs_clarification: bool = Field(
        description=(
            "Whether the user's first message will withhold required "
            "information, forcing the assistant to ask a clarifying "
            "question before any tool calls."
        ),
    )
    clarification_question: str = Field(
        default="",
        description=(
            "If needs_clarification is true, the question the assistant "
            "should ask to elicit the missing information. Empty string "
            "when needs_clarification is false."
        ),
    )
    withheld_parameters: list[str] = Field(
        default_factory=list,
        description=(
            "Parameter names the user will NOT mention in their first "
            "message. Used by the User agent to construct a deliberately "
            "vague opening line. Empty list when needs_clarification "
            "is false."
        ),
    )


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


PLANNER_SYSTEM_PROMPT = """\
You are a conversation scenario designer for synthetic training data.

Your job is to design ONE realistic user scenario that would naturally
motivate a specific sequence of tool calls. The scenario must feel like
something a real person would ask a helpful assistant to do — not a
contrived test case, not a generic placeholder.

Good personas are specific and grounded: "a mid-thirties consultant
flying to Berlin for a Tuesday client meeting" is useful; "a person who
needs information" is not.

You will respond by calling the `respond` tool with a JSON object
matching its schema.\
"""


def _format_chain_for_prompt(chain: SampledChain) -> str:
    """Render a chain as a human-readable numbered list for the prompt."""
    lines: list[str] = []
    for index, endpoint in enumerate(chain.endpoints, start=1):
        lines.append(f"Step {index}: {endpoint.id}")
        if endpoint.description:
            lines.append(f"  Purpose: {endpoint.description}")
        lines.append(f"  Category: {endpoint.category}")
        required = [parameter.name for parameter in endpoint.required_parameters]
        optional = [
            parameter.name
            for parameter in endpoint.parameters
            if not parameter.required
        ]
        if required:
            lines.append(f"  Required params: {', '.join(required)}")
        if optional:
            lines.append(f"  Optional params: {', '.join(optional)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_planner_prompt(
    chain: SampledChain, force_clarification: bool
) -> str:
    """Build the full user-turn prompt for the Planner."""
    chain_text = _format_chain_for_prompt(chain)

    if force_clarification:
        clarification_block = (
            "IMPORTANT: This scenario MUST require clarification. Set "
            "`needs_clarification` to true. Design the scenario so the "
            "user's first message is realistically vague — they omit "
            "one or two required parameters from step 1. Put those "
            "parameter names in `withheld_parameters`. Write a natural, "
            "conversational `clarification_question` that a helpful "
            "assistant would ask to get the missing information."
        )
    else:
        clarification_block = (
            "For this scenario, the user provides enough information "
            "in their first message to get started. Set "
            "`needs_clarification` to false, `clarification_question` "
            "to an empty string, and `withheld_parameters` to an empty "
            "list."
        )

    return f"""\
Design a realistic user scenario that would naturally require this \
sequence of tool calls:

{chain_text}

{clarification_block}

Guidelines:
- Write user_intent in first person ("I want to..." / "I need to...").
- Persona should be specific: age range, role, immediate context.
- The scenario must plausibly motivate ALL the tools in the sequence,
  in the order given.
- Do NOT reference the tool names directly. Real users don't know tool
  names; they describe what they want in ordinary language.

Respond with a JSON object matching the schema.\
"""


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


class PlannerAgent:
    """Turns a sampled tool chain into a ConversationPlan.

    Runs once per conversation. Uses the LLMClient's structured-output
    path, so the returned plan is guaranteed valid by the time the
    orchestrator sees it.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def plan(
        self,
        chain: SampledChain,
        force_clarification: bool = False,
    ) -> ConversationPlan:
        if chain.length == 0:
            raise ValueError("Cannot plan a conversation for an empty chain")

        prompt = _build_planner_prompt(chain, force_clarification)
        logger.debug("Planner prompt:\n%s", prompt)

        plan = self.llm.complete_structured(
            prompt=prompt,
            schema_model=ConversationPlan,
            system=PLANNER_SYSTEM_PROMPT,
            temperature=0.0,
            max_repair_attempts=2,
        )

        # Post-normalization: make the clarification fields consistent
        # with the orchestrator's decision. This is cheaper than a retry
        # and keeps downstream agents from having to double-check.
        if not force_clarification:
            plan = plan.model_copy(
                update={
                    "needs_clarification": False,
                    "clarification_question": "",
                    "withheld_parameters": [],
                }
            )
        elif not plan.needs_clarification:
            # Orchestrator flipped the clarification coin True but the LLM
            # still returned needs_clarification=False. Override so the
            # dataset hits the target clarification rate; pick a concrete
            # required parameter from the chain to withhold.
            logger.warning(
                "Planner returned needs_clarification=False despite "
                "force_clarification=True; overriding and picking first "
                "required param to withhold (chain %s)",
                [ep.id for ep in chain.endpoints],
            )
            first_required: str | None = None
            for ep in chain.endpoints:
                for p in ep.required_parameters:
                    first_required = p.name
                    break
                if first_required:
                    break
            plan = plan.model_copy(
                update={
                    "needs_clarification": True,
                    "clarification_question": plan.clarification_question
                    or (
                        f"Could you tell me the {first_required}?"
                        if first_required
                        else "Could you share a bit more detail?"
                    ),
                    "withheld_parameters": plan.withheld_parameters
                    or ([first_required] if first_required else []),
                }
            )

        return plan
