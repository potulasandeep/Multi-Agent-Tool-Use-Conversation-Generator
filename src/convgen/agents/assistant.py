"""
Assistant agent.

Runs on every turn after the first user message. Given the plan, the
conversation so far, the session store, and an orchestrator-supplied
ExpectedMode, it emits an AssistantAction describing what should happen
next: a clarification question, a tool call, or a final answer.

Design notes
------------
- The Assistant does NOT choose which tool to call. The chain was fixed
  at sample time in Step 5, and the orchestrator walks it in order. The
  Assistant fills in the *arguments*, not the endpoint. Forcing the
  endpoint via normalization is how we keep the generated dataset on
  the chain the sampler produced.

- Grounding is enforced by injecting the session store's available
  values into the prompt, with explicit "MUST use these values"
  language. We do not programmatically fill arguments — the whole point
  is to generate training data where the model learned to use the
  available values. If the LLM hallucinates, the judge in Step 12 will
  catch it and trigger repair.

- Post-normalization is load-bearing. The LLM routinely returns plans
  where `action=tool_call` but `final_answer` is also populated, or
  where the action enum doesn't match the orchestrator's mode. We
  override rather than retry: cheaper, more predictable, and leaves
  the LLM's creative fields (argument values, wording) untouched.

- Temperature is 0. The Assistant's job is constrained; creativity
  isn't helpful here. Dataset variety comes from chain diversity, not
  from this agent being random.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from convgen.agents.planner import ConversationPlan
from convgen.executor.mock import SessionStore
from convgen.llm.client import LLMClient
from convgen.registry.models import Endpoint

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

Message = dict[str, Any]
"""A single message in the conversation history.

Shape (role-dependent):
  {"role": "user", "content": "..."}
  {"role": "assistant", "action": "clarify", "content": "..."}
  {"role": "assistant", "action": "tool_call", "endpoint": "...", "arguments": {...}}
  {"role": "assistant", "action": "final_answer", "content": "..."}
  {"role": "tool", "endpoint": "...", "content": {...}}
"""


class ExpectedMode(str, Enum):
    """What the orchestrator expects the Assistant to do on this turn.

    The Assistant is told the mode explicitly and its output is
    normalized to match. This keeps chain execution deterministic.
    """

    CLARIFY = "clarify"
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"


# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------


class AssistantAction(BaseModel):
    """Structured output from the Assistant.

    Flat layout: easier for LLMs to produce reliably via structured
    output than nested objects. Irrelevant fields are empty strings or
    empty dicts, not null — this avoids nullable-schema quirks in some
    provider structured-output implementations.
    """

    action: ExpectedMode = Field(description="Which kind of turn this is.")
    clarification: str = Field(
        default="",
        description=(
            "If action=clarify, the question to ask the user. Empty "
            "otherwise."
        ),
    )
    tool_endpoint: str = Field(
        default="",
        description=(
            "If action=tool_call, the endpoint ID (e.g. 'hotels.book'). "
            "Empty otherwise."
        ),
    )
    tool_arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "If action=tool_call, the arguments dict for the endpoint. "
            "Empty dict otherwise. You MUST use values from the "
            "available session values list when a matching parameter "
            "exists."
        ),
    )
    final_answer: str = Field(
        default="",
        description=(
            "If action=final_answer, the wrap-up response to the user "
            "referencing the key results obtained. Empty otherwise."
        ),
    )


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


ASSISTANT_SYSTEM_PROMPT = """\
You are a helpful AI assistant accomplishing a user's task by calling
tools. You will be told on each turn what kind of action is expected:
clarify, tool_call, or final_answer. Produce exactly that kind of
action — do not mix them.

Critical rules for tool calls:

1. You MUST use values from the "Available values from previous tool
   calls" list whenever a parameter of the tool you are calling matches
   a key in that list. Inventing identifiers is a failure — if the
   available values contain a `hotel_id` and the tool you are calling
   needs a `hotel_id`, you MUST pick one from the list, not make one
   up.

2. If required information for a tool call is missing from both the
   conversation history and the available values list, something has
   gone wrong upstream — still produce your best attempt using the
   available information rather than refusing.

3. Do not mention tool names, endpoint IDs, or argument dicts to the
   user. Those are internal machinery. When clarifying or answering,
   speak in ordinary natural language.

You respond by calling the `respond` tool with a JSON object matching
its schema.\
"""


def _format_history(history: list[Message]) -> str:
    """Render the conversation history as a compact tagged transcript.

    Tags preserve the distinction between clarifications, tool calls,
    and final answers in prior assistant turns — important because the
    Assistant needs to understand what it said before without having
    to infer it from prose.
    """
    if not history:
        return "(no messages yet)"

    lines: list[str] = []
    for msg in history:
        role = msg.get("role")
        if role == "user":
            lines.append(f"User: {msg.get('content', '')}")
        elif role == "assistant":
            action = msg.get("action", "")
            if action == "clarify":
                lines.append(
                    f"Assistant [clarify]: {msg.get('content', '')}"
                )
            elif action == "tool_call":
                endpoint = msg.get("endpoint", "")
                args = msg.get("arguments", {})
                lines.append(
                    f"Assistant [tool_call {endpoint}]: "
                    f"{json.dumps(args, default=str)}"
                )
            elif action == "final_answer":
                lines.append(
                    f"Assistant [final_answer]: {msg.get('content', '')}"
                )
            else:
                lines.append(f"Assistant: {msg.get('content', '')}")
        elif role == "tool":
            endpoint = msg.get("endpoint", "")
            content = msg.get("content", {})
            summary = json.dumps(content, default=str)
            if len(summary) > 400:
                summary = summary[:400] + "... (truncated)"
            lines.append(f"Tool [{endpoint}]: {summary}")
        else:
            lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _format_session_values(session: SessionStore) -> str:
    """Render the session store as a labeled block for the prompt.

    Uses both raw keys ('hotel_id') and tool-namespaced keys
    ('hotels.hotel_id') as the executor records them.
    """
    values = session.available_values()
    if not values:
        return "(no values recorded yet)"
    lines: list[str] = []
    for key in sorted(values.keys()):
        rendered = json.dumps(values[key], default=str)
        lines.append(f"  {key}: {rendered}")
    return "\n".join(lines)


def _build_assistant_prompt(
    plan: ConversationPlan,
    history: list[Message],
    session: SessionStore,
    mode: ExpectedMode,
    next_endpoint: Endpoint | None,
    repair_hint: str = "",
) -> str:
    """Assemble the full Assistant prompt for one turn."""
    history_text = _format_history(history)
    session_text = _format_session_values(session)

    if mode == ExpectedMode.CLARIFY:
        directive = (
            "The user's first message was intentionally vague and is "
            "missing required information. Set action=clarify and ask "
            "a natural conversational question to elicit the missing "
            "information. Keep it friendly and specific. Do not call "
            "any tool yet."
        )
        if plan.clarification_question:
            directive += (
                f"\n\nSuggested question (you may rephrase): "
                f"{plan.clarification_question}"
            )
    elif mode == ExpectedMode.TOOL_CALL:
        if next_endpoint is None:
            raise ValueError("ExpectedMode.TOOL_CALL requires a next_endpoint")
        required = [p.name for p in next_endpoint.required_parameters]
        optional = [p.name for p in next_endpoint.parameters if not p.required]
        directive = (
            f"Set action=tool_call and call the endpoint "
            f"`{next_endpoint.id}`.\n"
            f"Set `tool_endpoint` to `{next_endpoint.id}`.\n"
            f"Required parameters: {required or '(none)'}\n"
            f"Optional parameters: {optional or '(none)'}\n\n"
            "Build `tool_arguments` as a JSON object mapping each "
            "required parameter (and relevant optional ones) to a "
            "value. You MUST use values from the Available values "
            "block above when the parameter matches a key there. "
            "For parameters that must be supplied from the user's "
            "request (like a city or a date they mentioned), extract "
            "them from the conversation history."
        )
    elif mode == ExpectedMode.FINAL_ANSWER:
        directive = (
            "All tool calls are complete. Set action=final_answer and "
            "write a short, natural wrap-up message to the user. "
            "Reference the key concrete results (like a confirmation "
            "ID or a booked item name) from the Available values block "
            "so the user knows their request was fulfilled. Do not "
            "mention tool names or internal IDs that aren't meaningful "
            "to the user."
        )
    else:
        raise ValueError(f"Unknown ExpectedMode: {mode}")

    repair_block = ""
    if repair_hint:
        repair_block = (
            f"\n## Repair guidance\n"
            f"{repair_hint}\n"
        )

    return f"""\
## User persona
{plan.persona}

## What the user wants
{plan.user_intent}

## Conversation so far
{history_text}

## Available values from previous tool calls
{session_text}
{repair_block}

## Your next action
{directive}

Respond with the `respond` tool.\
"""


# --------------------------------------------------------------------------
# Output normalization
# --------------------------------------------------------------------------


def _normalize_action(
    action: AssistantAction,
    mode: ExpectedMode,
    next_endpoint: Endpoint | None,
) -> AssistantAction:
    """Force the action to match the orchestrator's expected mode."""
    if action.action != mode:
        logger.warning(
            "Assistant returned action=%s but orchestrator expected %s; overriding",
            action.action,
            mode,
        )

    if mode == ExpectedMode.CLARIFY:
        return AssistantAction(
            action=ExpectedMode.CLARIFY,
            clarification=action.clarification or "Could you tell me more?",
        )

    if mode == ExpectedMode.TOOL_CALL:
        assert next_endpoint is not None
        if action.tool_endpoint and action.tool_endpoint != next_endpoint.id:
            logger.warning(
                "Assistant chose endpoint %s but orchestrator expected %s; overriding",
                action.tool_endpoint,
                next_endpoint.id,
            )
        return AssistantAction(
            action=ExpectedMode.TOOL_CALL,
            tool_endpoint=next_endpoint.id,
            tool_arguments=action.tool_arguments or {},
        )

    if mode == ExpectedMode.FINAL_ANSWER:
        return AssistantAction(
            action=ExpectedMode.FINAL_ANSWER,
            final_answer=action.final_answer or "Done.",
        )

    raise ValueError(f"Unknown mode: {mode}")


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


class AssistantAgent:
    """LLM-backed assistant that produces one action per turn."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def step(
        self,
        plan: ConversationPlan,
        history: list[Message],
        session: SessionStore,
        mode: ExpectedMode,
        next_endpoint: Endpoint | None = None,
        repair_hint: str = "",
    ) -> AssistantAction:
        if mode == ExpectedMode.TOOL_CALL and next_endpoint is None:
            raise ValueError("ExpectedMode.TOOL_CALL requires next_endpoint")

        prompt = _build_assistant_prompt(
            plan=plan,
            history=history,
            session=session,
            mode=mode,
            next_endpoint=next_endpoint,
            repair_hint=repair_hint,
        )
        logger.debug("Assistant prompt:\n%s", prompt)

        # OpenAI JSON-schema strict mode rejects open objects; tool_arguments
        # is dict[str, Any] (additionalProperties), so use non-strict here.
        raw_action = self.llm.complete_structured(
            prompt=prompt,
            schema_model=AssistantAction,
            system=ASSISTANT_SYSTEM_PROMPT,
            temperature=0.0,
            max_repair_attempts=2,
            strict_schema=False,
        )
        return _normalize_action(raw_action, mode, next_endpoint)
