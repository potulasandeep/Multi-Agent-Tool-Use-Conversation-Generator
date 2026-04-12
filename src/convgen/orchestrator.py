"""
Orchestrator — runs one end-to-end conversation from a sampled chain.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from pydantic import BaseModel, Field

from convgen.agents.assistant import AssistantAgent, ExpectedMode
from convgen.agents.planner import PlannerAgent
from convgen.agents.user import UserAgent
from convgen.executor.mock import ExecutorError, MockExecutor, SessionStore
from convgen.graph.sampler import SampledChain

logger = logging.getLogger(__name__)

# Separate RNG stream for the clarification coin so it is not coupled to
# any future use of `random.Random(seed)` inside the orchestrator.
_CLARIFICATION_RNG_SALT = 0xC1A41F1CA7104E


class Conversation(BaseModel):
    conversation_id: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]
    judge_scores: dict[str, Any] | None = Field(default=None)


def _internal_to_output(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in history:
        role = msg["role"]
        if role == "user":
            out.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            action = msg.get("action")
            if action == "tool_call":
                out.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "endpoint": msg["endpoint"],
                                "arguments": msg["arguments"],
                            }
                        ],
                    }
                )
            else:
                out.append({"role": "assistant", "content": msg.get("content", "")})
        elif role == "tool":
            out.append({"role": "tool", "content": msg["content"]})
        else:
            logger.warning("Unknown role in internal history: %s", role)
    return out


class Orchestrator:
    def __init__(
        self,
        planner: PlannerAgent,
        user: UserAgent,
        assistant: AssistantAgent,
        executor: MockExecutor,
        clarification_rate: float = 0.3,
    ) -> None:
        if not (0.0 <= clarification_rate <= 1.0):
            raise ValueError("clarification_rate must be in [0, 1]")
        self.planner = planner
        self.user = user
        self.assistant = assistant
        self.executor = executor
        self.clarification_rate = clarification_rate

    def run(
        self,
        chain: SampledChain,
        seed: int,
        conversation_id: str,
        repair_hint: str = "",
    ) -> Conversation:
        if chain.length == 0:
            raise ValueError("Cannot run a conversation for an empty chain")

        clar_rng = random.Random(seed ^ _CLARIFICATION_RNG_SALT)
        force_clarification = (
            clar_rng.random() < self.clarification_rate
        )

        plan = self.planner.plan(chain, force_clarification=force_clarification)
        session = SessionStore()
        history: list[dict[str, Any]] = []
        failed = False
        failure_reason: str | None = None

        history.append({"role": "user", "content": self.user.initial_message(plan)})

        if plan.needs_clarification:
            clar = self.assistant.step(
                plan=plan,
                history=history,
                session=session,
                mode=ExpectedMode.CLARIFY,
                repair_hint=repair_hint,
            )
            history.append(
                {"role": "assistant", "action": "clarify", "content": clar.clarification}
            )
            history.append(
                {
                    "role": "user",
                    "content": self.user.clarification_response(plan, history),
                }
            )

        for endpoint in chain.endpoints:
            action = self.assistant.step(
                plan=plan,
                history=history,
                session=session,
                mode=ExpectedMode.TOOL_CALL,
                next_endpoint=endpoint,
                repair_hint=repair_hint,
            )
            history.append(
                {
                    "role": "assistant",
                    "action": "tool_call",
                    "endpoint": action.tool_endpoint,
                    "arguments": action.tool_arguments,
                }
            )
            try:
                result = self.executor.execute(endpoint, action.tool_arguments, session)
            except ExecutorError as e:
                logger.warning("Executor failed for %s: %s", endpoint.id, e)
                failed = True
                failure_reason = f"executor_error: {e}"
                history.append(
                    {"role": "tool", "endpoint": endpoint.id, "content": {"error": str(e)}}
                )
                break
            history.append({"role": "tool", "endpoint": endpoint.id, "content": result})

        final = self.assistant.step(
            plan=plan,
            history=history,
            session=session,
            mode=ExpectedMode.FINAL_ANSWER,
            repair_hint=repair_hint,
        )
        history.append(
            {"role": "assistant", "action": "final_answer", "content": final.final_answer}
        )

        output_messages = _internal_to_output(history)
        endpoints_called = [
            m["endpoint"]
            for m in history
            if m["role"] == "assistant" and m.get("action") == "tool_call"
        ]
        tools_used = sorted({ep.split(".")[0] for ep in endpoints_called})

        metadata: dict[str, Any] = {
            "seed": seed,
            "chain_endpoint_ids": chain.endpoint_ids,
            "chain_length": chain.length,
            "pattern": chain.pattern,
            "tools_used": tools_used,
            "endpoints_called": endpoints_called,
            "num_turns": len(output_messages),
            "had_clarification": plan.needs_clarification,
            "force_clarification": force_clarification,
            "failed": failed,
            "failure_reason": failure_reason,
            "persona": plan.persona,
            "user_intent": plan.user_intent,
        }

        return Conversation(
            conversation_id=conversation_id,
            messages=output_messages,
            metadata=metadata,
        )
