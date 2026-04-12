"""
LLM-as-judge for generated conversations.

The Judge reads a completed Conversation and emits JudgeScores on four
dimensions: tool_correctness, grounding_fidelity, naturalness, and
task_completion. It also identifies the weakest turn so the repair
loop in Step 13 can do surgical fixes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from convgen.llm.client import LLMClient
from convgen.orchestrator import Conversation

logger = logging.getLogger(__name__)


class JudgeScores(BaseModel):
    tool_correctness: float = Field(ge=1.0, le=5.0)
    grounding_fidelity: float = Field(ge=1.0, le=5.0)
    naturalness: float = Field(ge=1.0, le=5.0)
    task_completion: float = Field(ge=1.0, le=5.0)
    reasoning: str = Field(min_length=10)
    failing_turn_index: int | None = None

    @property
    def mean(self) -> float:
        return (
            self.tool_correctness
            + self.grounding_fidelity
            + self.naturalness
            + self.task_completion
        ) / 4.0

    @property
    def min_score(self) -> float:
        return min(
            self.tool_correctness,
            self.grounding_fidelity,
            self.naturalness,
            self.task_completion,
        )


JUDGE_SYSTEM_PROMPT = """\
You are an expert rater of synthetic tool-use conversations, used to
train AI assistants that call APIs on users' behalf. You will read a
conversation and score it on four dimensions.

Scoring scale (for all four dimensions):
  5 = exemplary; a model trained on this would learn the right behavior
  4 = good; minor issues that do not mislead training
  3 = mediocre; usable with caveats, notable weaknesses
  2 = poor; would teach bad habits if used as training data
  1 = broken; fundamentally wrong

Half-point scores (like 4.5) are allowed for borderline cases.

## Dimension 1: tool_correctness

Are the tool calls structurally valid? Do they use the right endpoints,
include all required parameters, and pass sensible argument types?

  5 = Every tool call has a valid endpoint, all required parameters
      present, sensible types.
  2 = Tool calls have missing required parameters, wrong types, or
      the wrong endpoint for the stated task.
  1 = Tool calls are structurally broken or entirely absent when
      needed.

## Dimension 2: grounding_fidelity

This is the most important dimension. Do later tool calls use values
that were actually produced by earlier tool calls or earlier user
messages?

  5 = Every argument in every tool call either came from an earlier
      tool output in this conversation or was explicitly provided by
      the user. No invented identifiers.
  3 = One or two arguments look invented but most chain correctly.
  2 = Multiple arguments are clearly hallucinated IDs that were not
      in any prior tool output.
  1 = The chain is entirely disconnected; later tool calls ignore
      earlier tool outputs.

Pay close attention to ID-shaped fields (hotel_id, booking_id, user_id,
etc.). These are where hallucination usually happens.

Important: "grounded" means semantically appropriate, not just literally
present. If an ID appears in an earlier tool output but comes from a
different semantic domain than the tool now consuming it, that is a
grounding failure. Example: a song-lyrics endpoint receiving an ID that
was produced by a real-estate endpoint is NOT grounded, even though the
literal string came from a prior tool output. Score such cases 2 or
below regardless of their structural validity.

## Dimension 3: naturalness

Does the user+assistant dialogue read like a real interaction?

  5 = User messages sound like a real person; assistant replies are
      conversational and context-appropriate; no leakage of tool
      names or internal machinery.
  3 = Slightly stilted but understandable; occasional awkward phrasing.
  2 = Robotic, generic, or mentions tool/endpoint names to the user.
  1 = Unintelligible, repetitive, or obviously templated.

## Dimension 4: task_completion

Does the final assistant turn actually address the user's original
request?

  5 = The final turn clearly completes the task and references the
      concrete result (e.g., a confirmation ID or a booked item).
  3 = The task is partially addressed; some aspects are left hanging.
  2 = The final turn is generic without referencing the specific result.
  1 = The task was not completed; the conversation ended mid-way or
      with a non-sequitur.

## Reasoning and failing turn

After scoring, provide a short reasoning paragraph explaining any
score below 4. Then identify the single turn most responsible for the
lowest score as `failing_turn_index`, using the 0-based index into
the messages list. If all scores are 4 or higher, set
`failing_turn_index` to null.

You respond by calling the `respond` tool with a JSON object matching
its schema.\
"""


def _format_conversation_for_judge(conv: Conversation) -> str:
    lines: list[str] = []
    for i, msg in enumerate(conv.messages):
        role = msg.get("role")
        if role == "user":
            lines.append(f"[{i}] user: {msg.get('content', '')}")
        elif role == "assistant":
            if msg.get("content") is None and "tool_calls" in msg:
                calls = json.dumps(msg["tool_calls"], default=str)
                lines.append(f"[{i}] assistant (tool_call): {calls}")
            else:
                lines.append(f"[{i}] assistant: {msg.get('content', '')}")
        elif role == "tool":
            content = json.dumps(msg.get("content", {}), default=str)
            if len(content) > 500:
                content = content[:500] + "... (truncated)"
            lines.append(f"[{i}] tool: {content}")
        else:
            lines.append(f"[{i}] {role}: {msg}")
    return "\n".join(lines)


def _build_judge_prompt(conv: Conversation) -> str:
    transcript = _format_conversation_for_judge(conv)
    md = conv.metadata
    intent = md.get("user_intent", "(unknown)")
    chain = md.get("chain_endpoint_ids", [])
    return f"""\
## Intended scenario
User intent: {intent}
Intended tool chain (from the sampler): {chain}

## Conversation transcript (indexed)
{transcript}

Score this conversation on all four dimensions. Respond with the
`respond` tool.\
"""


class Judge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def score(self, conv: Conversation) -> JudgeScores:
        if not conv.messages:
            raise ValueError("Cannot score a conversation with no messages")
        prompt = _build_judge_prompt(conv)
        logger.debug("Judge prompt:\n%s", prompt)
        scores = self.llm.complete_structured(
            prompt=prompt,
            schema_model=JudgeScores,
            system=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_repair_attempts=2,
        )
        if scores.failing_turn_index is not None:
            if not (0 <= scores.failing_turn_index < len(conv.messages)):
                logger.warning("Judge returned out-of-range failing_turn_index; clearing")
                scores = scores.model_copy(update={"failing_turn_index": None})
        return scores

    def score_and_attach(self, conv: Conversation) -> Conversation:
        scores = self.score(conv)
        return conv.model_copy(
            update={"judge_scores": self.scores_to_dict(scores)}
        )

    @staticmethod
    def scores_to_dict(scores: JudgeScores) -> dict[str, Any]:
        return {
            "tool_correctness": scores.tool_correctness,
            "grounding_fidelity": scores.grounding_fidelity,
            "naturalness": scores.naturalness,
            "task_completion": scores.task_completion,
            "mean": scores.mean,
            "min_score": scores.min_score,
            "reasoning": scores.reasoning,
            "failing_turn_index": scores.failing_turn_index,
        }
