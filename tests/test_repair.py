"""Tests for the RepairLoop and repair_hint plumbing."""

from __future__ import annotations

from dataclasses import dataclass

from convgen.graph.sampler import SampledChain
from convgen.judge.judge import JudgeScores
from convgen.judge.repair import RepairConfig, RepairLoop
from convgen.orchestrator import Conversation
from convgen.registry.models import Endpoint, Parameter, ResponseField


def _ep(endpoint_id: str, *, tool: str = "toolX", category: str = "Cat") -> Endpoint:
    name = endpoint_id.split(".", 1)[1]
    return Endpoint(
        id=endpoint_id,
        tool_name=tool,
        name=name,
        category=category,
        parameters=[Parameter(name="x", type="string", required=True)],
        response_fields=[ResponseField(name="y", type="string")],
    )


class FakeOrchestrator:
    def __init__(self) -> None:
        self.hints: list[str] = []

    def run(
        self,
        chain: SampledChain,
        seed: int,
        conversation_id: str,
        repair_hint: str = "",
    ) -> Conversation:
        # Identify which attempt produced the returned conversation.
        attempt_index = len(self.hints)
        self.hints.append(repair_hint)
        return Conversation(
            conversation_id=conversation_id,
            messages=[{"role": "user", "content": "hi"}],
            metadata={
                "seed": seed,
                "attempt_index": attempt_index,
                "repair_hint": repair_hint,
                "chain_endpoint_ids": chain.endpoint_ids,
                "chain_length": chain.length,
                "pattern": chain.pattern,
            },
        )


@dataclass
class FakeJudge:
    scores: list[JudgeScores]

    def score(self, conversation: Conversation) -> JudgeScores:
        return self.scores.pop(0)


def _scores(
    *,
    tool: float,
    grounding: float,
    natural: float,
    completion: float,
    failing_turn_index: int | None = None,
    reasoning: str = "Reasoning is intentionally verbose enough.",
) -> JudgeScores:
    return JudgeScores(
        tool_correctness=tool,
        grounding_fidelity=grounding,
        naturalness=natural,
        task_completion=completion,
        reasoning=reasoning,
        failing_turn_index=failing_turn_index,
    )


def test_repair_loop_returns_best_across_attempts() -> None:
    chain = SampledChain(
        endpoints=[_ep("toolA.one"), _ep("toolB.two")],
        pattern="sequential",
    )

    # attempt 0: fails threshold
    # attempt 1: passes (but worse than attempt 2)
    # attempt 2: passes and is best
    scores = [
        _scores(tool=5.0, grounding=2.0, natural=4.0, completion=2.0, failing_turn_index=3),
        _scores(tool=5.0, grounding=3.0, natural=4.0, completion=3.0, failing_turn_index=3),
        _scores(tool=5.0, grounding=4.0, natural=5.0, completion=2.0, failing_turn_index=4),
    ]

    orch = FakeOrchestrator()
    judge = FakeJudge(scores=scores)
    loop = RepairLoop(
        orchestrator=orch, judge=judge, config=RepairConfig(max_repairs=2)
    )
    out = loop.run(chain=chain, seed=123, conversation_id="c1")

    assert out.metadata["was_repaired"] is True
    assert out.metadata["repair_attempts"] == 3
    assert out.metadata["attempt_index"] == 2  # best should be final attempt

    # First orchestrator call has no repair hint.
    assert orch.hints[0] == ""
    assert "Repair attempt 1:" in orch.hints[1]
    assert "Repair attempt 2:" in orch.hints[2]
    assert "failing turn index" in orch.hints[1].lower()


def test_repair_loop_when_all_fail_returns_best() -> None:
    chain = SampledChain(
        endpoints=[_ep("toolA.one"), _ep("toolA.two")],
        pattern="sequential",
    )

    # All attempts fail due to mean/min thresholds, but attempt 2 is best.
    scores = [
        _scores(tool=2.0, grounding=2.0, natural=2.0, completion=2.0, failing_turn_index=1),
        _scores(tool=3.0, grounding=2.5, natural=3.0, completion=3.0, failing_turn_index=2),
        _scores(tool=4.0, grounding=2.6, natural=3.0, completion=2.8, failing_turn_index=3),
    ]

    orch = FakeOrchestrator()
    judge = FakeJudge(scores=scores)
    loop = RepairLoop(
        orchestrator=orch, judge=judge, config=RepairConfig(max_repairs=2)
    )
    out = loop.run(chain=chain, seed=123, conversation_id="c2")

    assert out.metadata["was_repaired"] is True
    assert out.metadata["repair_attempts"] == 3
    assert out.metadata["attempt_index"] == 2  # best among failed attempts

    # Sanity check: judge_scores attached.
    assert out.judge_scores is not None
    assert out.judge_scores["min_score"] >= 2.5

