"""
Repair loop for low-scoring conversations.

This module implements a "soft" repair strategy:
  - Generate a conversation once.
  - Ask the Judge to score it.
  - If the score is below threshold, generate additional conversations by
    re-running the orchestrator with a `repair_hint` injected into the
    Assistant prompt.
  - Keep the highest-scoring attempt across all generated repair attempts.

The repair hint includes an attempt index so each attempt has a distinct
prompt (avoiding prompt-level cache collisions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from convgen.graph.sampler import SampledChain
from convgen.judge.judge import Judge, JudgeScores
from convgen.orchestrator import Conversation, Orchestrator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepairConfig:
    """Tunable parameters for the repair loop."""

    threshold: float = 3.4
    min_threshold: float = 2.5
    max_repairs: int = 2


class RepairLoop:
    """Generate and repair low-scoring conversations."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        judge: Judge,
        config: RepairConfig | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.judge = judge
        self.config = config or RepairConfig()

    def _passes(self, scores: JudgeScores) -> bool:
        """Two-sided acceptance gate: mean and min-dimension must both pass."""
        return (
            scores.mean >= self.config.threshold
            and scores.min_score >= self.config.min_threshold
        )

    def _build_repair_hint(
        self, scores: JudgeScores, *, repair_attempt_number: int
    ) -> str:
        """Build a critique string injected into the Assistant prompt."""
        return (
            f"Repair attempt {repair_attempt_number}:\n"
            f"Judge scores: tool_correctness={scores.tool_correctness:.1f}, "
            f"grounding_fidelity={scores.grounding_fidelity:.1f}, "
            f"naturalness={scores.naturalness:.1f}, "
            f"task_completion={scores.task_completion:.1f}\n"
            f"Mean={scores.mean:.2f}, min_score={scores.min_score:.2f}\n"
            f"Judge reasoning: {scores.reasoning}\n"
            f"Failing turn index: {scores.failing_turn_index}\n\n"
            "Please revise your next action(s) to fix the issues above. "
            "Prioritize grounding_fidelity by only using values from the "
            "Available values block when a matching parameter exists."
        )

    def run(
        self, chain: SampledChain, seed: int, conversation_id: str
    ) -> Conversation:
        """Run the repair loop and return the best attempt."""

        # Initial attempt ---------------------------------------------------
        initial_convo = self.orchestrator.run(
            chain=chain,
            seed=seed,
            conversation_id=conversation_id,
        )
        initial_scores = self.judge.score(initial_convo)

        if self._passes(initial_scores):
            logger.debug(
                "Conversation %s passed on initial attempt", conversation_id
            )
            return initial_convo.model_copy(
                update={
                    "judge_scores": Judge.scores_to_dict(initial_scores),
                    "metadata": {
                        **initial_convo.metadata,
                        "repair_attempts": 1,
                        "was_repaired": False,
                    },
                }
            )

        # Repair attempts ---------------------------------------------------
        best_convo: Conversation = initial_convo
        best_scores: JudgeScores = initial_scores

        repair_attempts_executed = 1
        current_scores = initial_scores

        for i in range(self.config.max_repairs):
            repair_attempt_number = i + 1
            hint = self._build_repair_hint(
                current_scores,
                repair_attempt_number=repair_attempt_number,
            )
            repair_convo = self.orchestrator.run(
                chain=chain,
                seed=seed,
                conversation_id=conversation_id,
                repair_hint=hint,
            )
            repair_scores = self.judge.score(repair_convo)
            repair_attempts_executed += 1
            current_scores = repair_scores

            # Keep the best-scoring attempt across all repair calls.
            is_better = (repair_scores.mean, repair_scores.min_score) > (
                best_scores.mean,
                best_scores.min_score,
            )
            if is_better:
                best_convo = repair_convo
                best_scores = repair_scores

            logger.info(
                "Repair %s for %s: mean=%.2f min=%.2f (best mean=%.2f)",
                repair_attempt_number,
                conversation_id,
                repair_scores.mean,
                repair_scores.min_score,
                best_scores.mean,
            )

        return best_convo.model_copy(
            update={
                "judge_scores": Judge.scores_to_dict(best_scores),
                "metadata": {
                    **best_convo.metadata,
                    "repair_attempts": repair_attempts_executed,
                    "was_repaired": True,
                },
            }
        )

