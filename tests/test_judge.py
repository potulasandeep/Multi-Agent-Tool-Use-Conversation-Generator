"""Tests for the judge. FakeLLMClient throughout — no network."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from convgen.judge.judge import (
    JUDGE_SYSTEM_PROMPT,
    Judge,
    JudgeScores,
    _build_judge_prompt,
    _format_conversation_for_judge,
)
from convgen.llm import FakeLLMClient
from convgen.orchestrator import Conversation


# ---------------------- fixture helpers ----------------------


def _good_convo() -> Conversation:
    return Conversation(
        conversation_id="c_good",
        messages=[
            {"role": "user", "content": "Find me a hotel in Paris"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "endpoint": "hotels.search",
                        "arguments": {"city": "Paris"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": {
                    "results": [
                        {"id": "hot_1234", "name": "Hotel du Marais"}
                    ]
                },
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "endpoint": "hotels.book",
                        "arguments": {"hotel_id": "hot_1234"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": {
                    "booking_id": "bk_9999",
                    "status": "confirmed",
                },
            },
            {
                "role": "assistant",
                "content": "Booked! Your confirmation is bk_9999.",
            },
        ],
        metadata={
            "seed": 1,
            "chain_endpoint_ids": ["hotels.search", "hotels.book"],
            "chain_length": 2,
            "pattern": "sequential",
            "tools_used": ["hotels"],
            "endpoints_called": ["hotels.search", "hotels.book"],
            "num_turns": 6,
            "had_clarification": False,
            "force_clarification": False,
            "failed": False,
            "failure_reason": None,
            "persona": "A traveler",
            "user_intent": "Book a hotel in Paris",
        },
    )


def _good_scores_dict(**overrides) -> dict:
    base = {
        "tool_correctness": 5.0,
        "grounding_fidelity": 5.0,
        "naturalness": 4.5,
        "task_completion": 5.0,
        "reasoning": "All tool calls valid, IDs correctly chained.",
        "failing_turn_index": None,
    }
    base.update(overrides)
    return base


# ---------------------- JudgeScores schema ----------------------


class TestJudgeScoresSchema:
    def test_valid_scores(self):
        scores = JudgeScores(**_good_scores_dict())
        assert scores.tool_correctness == 5.0
        assert scores.reasoning.startswith("All tool calls valid")

    def test_score_below_one_rejected(self):
        with pytest.raises(ValidationError):
            JudgeScores(**_good_scores_dict(tool_correctness=0.5))

    def test_score_above_five_rejected(self):
        with pytest.raises(ValidationError):
            JudgeScores(**_good_scores_dict(naturalness=5.5))

    def test_short_reasoning_rejected(self):
        with pytest.raises(ValidationError):
            JudgeScores(**_good_scores_dict(reasoning="too short"))

    def test_half_point_scores_allowed(self):
        scores = JudgeScores(**_good_scores_dict(naturalness=3.5))
        assert scores.naturalness == 3.5

    def test_mean_property(self):
        scores = JudgeScores(
            **_good_scores_dict(
                tool_correctness=4.0,
                grounding_fidelity=4.0,
                naturalness=4.0,
                task_completion=4.0,
            )
        )
        assert scores.mean == 4.0

    def test_min_score_property(self):
        scores = JudgeScores(
            **_good_scores_dict(
                tool_correctness=5.0,
                grounding_fidelity=2.0,
                naturalness=4.5,
                task_completion=5.0,
            )
        )
        assert scores.min_score == 2.0


# ---------------------- conversation formatting ----------------------


class TestFormatConversation:
    def test_indices_are_zero_based(self):
        out = _format_conversation_for_judge(_good_convo())
        assert out.startswith("[0] user:")
        assert "[1] assistant" in out

    def test_tool_call_rendered_inline(self):
        out = _format_conversation_for_judge(_good_convo())
        assert "hotels.search" in out
        assert "Paris" in out

    def test_tool_content_rendered_as_json(self):
        out = _format_conversation_for_judge(_good_convo())
        assert "hot_1234" in out
        assert "Hotel du Marais" in out

    def test_long_tool_output_truncated(self):
        convo = _good_convo()
        convo.messages.append({"role": "tool", "content": {"blob": "x" * 2000}})
        out = _format_conversation_for_judge(convo)
        assert "truncated" in out


# ---------------------- prompt building ----------------------


class TestBuildPrompt:
    def test_prompt_includes_intent_and_chain(self):
        prompt = _build_judge_prompt(_good_convo())
        assert "Book a hotel in Paris" in prompt
        assert "hotels.search" in prompt
        assert "hotels.book" in prompt

    def test_prompt_includes_transcript(self):
        prompt = _build_judge_prompt(_good_convo())
        assert "[0] user:" in prompt
        assert "Hotel du Marais" in prompt

    def test_prompt_reiterates_grounding_priority(self):
        _build_judge_prompt(_good_convo())
        assert "grounding_fidelity" in JUDGE_SYSTEM_PROMPT


# ---------------------- Judge.score ----------------------


class TestJudgeScore:
    def test_happy_path(self):
        llm = FakeLLMClient(default_structured=_good_scores_dict())
        judge = Judge(llm)
        scores = judge.score(_good_convo())
        assert isinstance(scores, JudgeScores)
        assert scores.tool_correctness == 5.0
        assert scores.failing_turn_index is None

    def test_system_prompt_used(self):
        llm = FakeLLMClient(default_structured=_good_scores_dict())
        judge = Judge(llm)
        judge.score(_good_convo())
        assert llm.calls[0]["system"] == JUDGE_SYSTEM_PROMPT

    def test_structured_call(self):
        llm = FakeLLMClient(default_structured=_good_scores_dict())
        judge = Judge(llm)
        judge.score(_good_convo())
        assert llm.calls[0]["kind"] == "structured"

    def test_empty_conversation_rejected(self):
        llm = FakeLLMClient(default_structured=_good_scores_dict())
        judge = Judge(llm)
        convo = Conversation(conversation_id="empty", messages=[], metadata={})
        with pytest.raises(ValueError, match="no messages"):
            judge.score(convo)

    def test_out_of_range_failing_turn_cleared(self, caplog: pytest.LogCaptureFixture):
        llm = FakeLLMClient(
            default_structured=_good_scores_dict(
                tool_correctness=2.0,
                reasoning="Turn 3 was clearly incorrect.",
                failing_turn_index=999,
            )
        )
        judge = Judge(llm)
        with caplog.at_level(logging.WARNING):
            scores = judge.score(_good_convo())
        assert scores.failing_turn_index is None
        assert any("out-of-range" in r.message for r in caplog.records)

    def test_in_range_failing_turn_preserved(self):
        llm = FakeLLMClient(
            default_structured=_good_scores_dict(
                naturalness=2.0,
                reasoning="Turn 5 was robotic.",
                failing_turn_index=5,
            )
        )
        judge = Judge(llm)
        scores = judge.score(_good_convo())
        assert scores.failing_turn_index == 5


# ---------------------- Judge.score_and_attach ----------------------


class TestScoreAndAttach:
    def test_attaches_scores_dict(self):
        llm = FakeLLMClient(default_structured=_good_scores_dict())
        judge = Judge(llm)
        scored = judge.score_and_attach(_good_convo())
        assert scored.judge_scores is not None
        assert scored.judge_scores["tool_correctness"] == 5.0
        assert scored.judge_scores["grounding_fidelity"] == 5.0
        assert scored.judge_scores["naturalness"] == 4.5
        assert scored.judge_scores["task_completion"] == 5.0
        assert "mean" in scored.judge_scores
        assert "min_score" in scored.judge_scores
        assert "reasoning" in scored.judge_scores
        assert "failing_turn_index" in scored.judge_scores

    def test_original_conversation_not_mutated(self):
        llm = FakeLLMClient(default_structured=_good_scores_dict())
        judge = Judge(llm)
        convo = _good_convo()
        _ = judge.score_and_attach(convo)
        assert convo.judge_scores is None

    def test_mean_in_dict_matches_property(self):
        llm = FakeLLMClient(
            default_structured=_good_scores_dict(
                tool_correctness=5.0,
                grounding_fidelity=3.0,
                naturalness=4.0,
                task_completion=4.0,
            )
        )
        judge = Judge(llm)
        scored = judge.score_and_attach(_good_convo())
        assert scored.judge_scores["mean"] == 4.0
        assert scored.judge_scores["min_score"] == 3.0


# ---------------------- low-score scenarios ----------------------


class TestLowScoreScenarios:
    def test_hallucinated_id_scored_low_grounding(self):
        llm = FakeLLMClient(
            default_structured=_good_scores_dict(
                grounding_fidelity=2.0,
                reasoning=(
                    "Turn 3 uses hotel_id='hot_9999' which was never "
                    "returned by any prior tool call."
                ),
                failing_turn_index=3,
            )
        )
        judge = Judge(llm)
        scores = judge.score(_good_convo())
        assert scores.grounding_fidelity == 2.0
        assert scores.min_score == 2.0
        assert scores.failing_turn_index == 3

    def test_incomplete_task_scored_low_completion(self):
        llm = FakeLLMClient(
            default_structured=_good_scores_dict(
                task_completion=2.0,
                reasoning="Final turn did not reference any booking.",
                failing_turn_index=5,
            )
        )
        judge = Judge(llm)
        scored = judge.score_and_attach(_good_convo())
        assert scored.judge_scores["task_completion"] == 2.0
        assert scored.judge_scores["failing_turn_index"] == 5
