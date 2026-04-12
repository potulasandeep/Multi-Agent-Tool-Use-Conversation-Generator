"""
User agent.

The only free-text agent in the pipeline. Plays the human user in a
generated conversation. Has two modes:

  initial_message(plan)
      The first user turn. If the plan has withheld parameters, the
      message is deliberately vague about them, forcing the assistant
      to ask a clarifying question.

  clarification_response(plan, history)
      A follow-up user turn after the assistant asked for clarification.
      The user supplies the previously withheld info in natural prose.

Design notes
------------
Temperature is 0.7: user messages should vary naturally across
conversations. Reproducibility on reruns is handled by the LLM cache,
which keys on (prompt, temperature). The first run is non-deterministic;
subsequent runs with the same seed and cache hit the same values.

No structured output. The User's output is just the `content` of a
user turn — there is nothing to parse. Quality (naturalness, did the
user actually answer the question) is checked by the Judge in Step 12,
not by this agent.
"""

from __future__ import annotations

import logging

from convgen.agents.planner import ConversationPlan
from convgen.llm.client import LLMClient

logger = logging.getLogger(__name__)

Message = dict


USER_SYSTEM_PROMPT = """\
You are playing a human user talking to an AI assistant. Respond as the
user would — in natural, conversational, first-person language.

Hard rules:
- NEVER mention tool names, endpoint IDs, parameter names, API terms,
  or anything that sounds like internal machinery. Real users don't
  know these things.
- NEVER say things like "please call the search tool" or "use the
  booking endpoint." Speak in ordinary human terms about what you
  want.
- Keep messages short: one to three sentences is ideal. Real users are
  not verbose when talking to assistants.
- Do not use markdown, bullet points, or formatting. Plain prose only.
- Do not break character. You ARE the user described in the persona;
  you are not an AI pretending to be one.\
"""


def _build_initial_prompt(plan: ConversationPlan) -> str:
    """Prompt for the user's very first message.

    If the plan has withheld parameters, instruct the LLM to be
    deliberately vague about them so the assistant has a reason to ask
    a clarifying question.
    """
    if plan.needs_clarification and plan.withheld_parameters:
        withholding = (
            f"\n\nIMPORTANT: In your opening message, deliberately do "
            f"NOT mention the following details, even though a real "
            f"user planning this task would eventually need to provide "
            f"them: {', '.join(plan.withheld_parameters)}.\n"
            f"Your opening should sound like a natural, slightly vague "
            f"opener that leaves these details unsaid. A helpful "
            f"assistant will then ask about them."
        )
    else:
        withholding = (
            "\n\nInclude enough concrete information in your opening "
            "message that a helpful assistant could reasonably start "
            "working on your request without further clarification."
        )

    return f"""\
## Your persona
{plan.persona}

## What you want
{plan.user_intent}

## Your task
Write the OPENING message of a conversation with an AI assistant. This \
is the very first thing you say. Keep it short (1-3 sentences) and \
natural. Speak as yourself, in first person.{withholding}\
"""


def _build_clarification_response_prompt(
    plan: ConversationPlan, history: list[Message]
) -> str:
    """Prompt for the user's response to a clarification question."""
    clarification_question = ""
    for msg in reversed(history):
        if msg.get("role") == "assistant" and msg.get("action") == "clarify":
            clarification_question = msg.get("content", "")
            break

    withheld = plan.withheld_parameters or []
    withheld_text = ", ".join(withheld) if withheld else "(no specific fields recorded)"

    return f"""\
## Your persona
{plan.persona}

## What you want
{plan.user_intent}

## The assistant just asked
"{clarification_question}"

## Your task
Write a short (1-2 sentences) natural response answering the \
assistant's question. Make sure to supply values for the following \
details you omitted earlier: {withheld_text}. Invent reasonable \
values consistent with your persona — there is no "correct" answer. \
Respond in plain first-person prose.\
"""


class UserAgent:
    """Generates user turns as free-text strings."""

    def __init__(self, llm: LLMClient, temperature: float = 0.7) -> None:
        self.llm = llm
        self.temperature = temperature

    def initial_message(self, plan: ConversationPlan) -> str:
        prompt = _build_initial_prompt(plan)
        logger.debug("User initial prompt:\n%s", prompt)
        text = self.llm.complete(
            prompt=prompt,
            system=USER_SYSTEM_PROMPT,
            temperature=self.temperature,
        )
        return text.strip()

    def clarification_response(
        self, plan: ConversationPlan, history: list[Message]
    ) -> str:
        if not plan.needs_clarification:
            raise ValueError(
                "clarification_response called but plan.needs_clarification is False"
            )
        prompt = _build_clarification_response_prompt(plan, history)
        logger.debug("User clarification-response prompt:\n%s", prompt)
        text = self.llm.complete(
            prompt=prompt,
            system=USER_SYSTEM_PROMPT,
            temperature=self.temperature,
        )
        return text.strip()
