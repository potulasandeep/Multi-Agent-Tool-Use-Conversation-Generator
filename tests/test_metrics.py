"""Tests for diversity metrics."""

from __future__ import annotations

import math

from convgen.orchestrator import Conversation
from convgen.steering.metrics import (
    category_gini,
    clarification_rate,
    mean_judge_scores,
    multi_step_ratio,
    multi_tool_ratio,
    tool_pair_entropy,
    unique_tool_coverage,
)


def _conv(
    cid: str = "c",
    endpoints: list[str] | None = None,
    tools: list[str] | None = None,
    had_clarification: bool = False,
    judge_scores: dict | None = None,
) -> Conversation:
    return Conversation(
        conversation_id=cid,
        messages=[{"role": "user", "content": "hi"}],
        metadata={
            "endpoints_called": endpoints or [],
            "tools_used": tools or [],
            "had_clarification": had_clarification,
        },
        judge_scores=judge_scores,
    )


class TestUniqueToolCoverage:
    def test_empty(self):
        assert unique_tool_coverage([]) == 0.0

    def test_raw_count(self):
        convs = [
            _conv(tools=["hotels"]),
            _conv(tools=["flights"]),
            _conv(tools=["hotels"]),
        ]
        assert unique_tool_coverage(convs) == 2.0

    def test_fraction(self):
        convs = [_conv(tools=["hotels"]), _conv(tools=["flights"])]
        assert unique_tool_coverage(convs, total_available_tools=4) == 0.5

    def test_zero_total_returns_zero(self):
        convs = [_conv(tools=["x"])]
        assert unique_tool_coverage(convs, total_available_tools=0) == 0.0


class TestPairEntropy:
    def test_empty_corpus(self):
        assert tool_pair_entropy([]) == 0.0

    def test_single_pair(self):
        convs = [_conv(endpoints=["a.x", "b.y"])]
        assert tool_pair_entropy(convs) == 0.0

    def test_two_distinct_pairs(self):
        convs = [
            _conv(endpoints=["a.x", "b.y"]),
            _conv(endpoints=["c.x", "d.y"]),
        ]
        assert math.isclose(tool_pair_entropy(convs), 1.0)

    def test_skewed_distribution_lower_entropy(self):
        convs = [
            _conv(endpoints=["a.x", "b.y"]),
            _conv(endpoints=["a.x", "b.y"]),
            _conv(endpoints=["a.x", "b.y"]),
            _conv(endpoints=["c.x", "d.y"]),
        ]
        ent = tool_pair_entropy(convs)
        assert 0.0 < ent < 1.0


class TestCategoryGini:
    def test_empty(self):
        assert category_gini([]) == 0.0

    def test_perfectly_balanced(self):
        convs = [
            _conv(tools=["hotels"]),
            _conv(tools=["flights"]),
            _conv(tools=["weather"]),
        ]
        assert category_gini(convs) == 0.0

    def test_concentrated_is_higher(self):
        balanced = [_conv(tools=["a"]), _conv(tools=["b"]), _conv(tools=["c"])]
        skewed = [
            _conv(tools=["a"]),
            _conv(tools=["a"]),
            _conv(tools=["a"]),
            _conv(tools=["b"]),
        ]
        assert category_gini(skewed) > category_gini(balanced)


class TestRatios:
    def test_multi_step_ratio(self):
        convs = [
            _conv(endpoints=["a.x", "b.y", "c.z"]),
            _conv(endpoints=["a.x", "b.y", "c.z", "d.w"]),
            _conv(endpoints=["a.x"]),
        ]
        assert multi_step_ratio(convs, min_steps=3) == 2 / 3

    def test_multi_tool_ratio(self):
        convs = [
            _conv(tools=["hotels", "flights"]),
            _conv(tools=["hotels"]),
        ]
        assert multi_tool_ratio(convs, min_tools=2) == 0.5

    def test_clarification_rate(self):
        convs = [
            _conv(had_clarification=True),
            _conv(had_clarification=False),
            _conv(had_clarification=True),
            _conv(had_clarification=False),
        ]
        assert clarification_rate(convs) == 0.5

    def test_empty_returns_zero(self):
        assert multi_step_ratio([]) == 0.0
        assert multi_tool_ratio([]) == 0.0
        assert clarification_rate([]) == 0.0


class TestMeanJudgeScores:
    def test_empty_returns_empty(self):
        assert mean_judge_scores([]) == {}

    def test_skips_unscored(self):
        convs = [
            _conv(judge_scores=None),
            _conv(judge_scores={"tool_correctness": 4.0, "mean": 4.0}),
        ]
        out = mean_judge_scores(convs)
        assert out["tool_correctness"] == 4.0
        assert out["mean"] == 4.0

    def test_averages_correctly(self):
        convs = [
            _conv(judge_scores={"tool_correctness": 5.0, "mean": 4.5}),
            _conv(judge_scores={"tool_correctness": 3.0, "mean": 3.5}),
        ]
        out = mean_judge_scores(convs)
        assert out["tool_correctness"] == 4.0
        assert out["mean"] == 4.0

