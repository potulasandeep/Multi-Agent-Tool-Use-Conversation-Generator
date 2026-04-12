"""
Diversity metrics for generated conversation datasets.

These metrics are designed to measure exactly what cross-conversation
steering is supposed to change: how broadly the corpus covers the
available tools, how varied its tool combinations are, and how balanced
its category distribution is. The diversity experiment (Step 15) will
compute these on Run A (no steering) and Run B (steering) to quantify
the steering effect.

All functions take a list of Conversation objects (in-memory) so they
are pure and unit-testable. The CLI loads JSONL into Conversations
before calling them.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from convgen.orchestrator import Conversation


def unique_tool_coverage(
    conversations: Iterable[Conversation],
    total_available_tools: int | None = None,
) -> float:
    """Fraction of available tools that appear at least once in the corpus.

    If `total_available_tools` is None, returns the raw count of unique
    tools instead of a fraction. Pass the registry size from the CLI
    to get a [0, 1] coverage score.
    """
    seen: set[str] = set()
    for conv in conversations:
        for tool in conv.metadata.get("tools_used", []):
            seen.add(tool)
    if total_available_tools is None:
        return float(len(seen))
    if total_available_tools <= 0:
        return 0.0
    return len(seen) / total_available_tools


def tool_pair_entropy(conversations: Iterable[Conversation]) -> float:
    """Shannon entropy (bits) over adjacent tool pairs across all chains.

    For each conversation, walks `endpoints_called` and emits adjacent
    (endpoint_i, endpoint_{i+1}) pairs. Higher entropy = more varied
    combinations. Returns 0.0 for an empty corpus or chains with no
    pairs.

    This is the most discriminating diversity metric: two corpora can
    have the same unique-tool coverage but very different pair entropy
    if one keeps reusing the same combinations.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for conv in conversations:
        endpoints = conv.metadata.get("endpoints_called", [])
        for a, b in zip(endpoints, endpoints[1:]):
            pairs[(a, b)] += 1
    total = sum(pairs.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in pairs.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def category_gini(conversations: Iterable[Conversation]) -> float:
    """Gini coefficient over category usage counts.

    0.0 means perfectly balanced (every category used equally), 1.0
    means perfectly concentrated (one category dominates everything).
    Lower is better when measuring diversity.
    """
    counts: Counter[str] = Counter()
    for conv in conversations:
        for tool in conv.metadata.get("tools_used", []):
            counts[tool] += 1

    values = sorted(counts.values())
    n = len(values)
    if n == 0:
        return 0.0
    if all(v == values[0] for v in values):
        return 0.0
    weighted_sum = 0
    for i, v in enumerate(values, start=1):
        weighted_sum += i * v
    total = sum(values)
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def multi_step_ratio(
    conversations: Iterable[Conversation], min_steps: int = 3
) -> float:
    """Fraction of conversations with at least `min_steps` tool calls."""
    convs = list(conversations)
    if not convs:
        return 0.0
    multi = sum(
        1
        for c in convs
        if len(c.metadata.get("endpoints_called", [])) >= min_steps
    )
    return multi / len(convs)


def multi_tool_ratio(
    conversations: Iterable[Conversation], min_tools: int = 2
) -> float:
    """Fraction of conversations using at least `min_tools` distinct tools."""
    convs = list(conversations)
    if not convs:
        return 0.0
    multi = sum(
        1
        for c in convs
        if len(set(c.metadata.get("tools_used", []))) >= min_tools
    )
    return multi / len(convs)


def clarification_rate(conversations: Iterable[Conversation]) -> float:
    """Fraction of conversations that included a clarification turn."""
    convs = list(conversations)
    if not convs:
        return 0.0
    clar = sum(
        1 for c in convs if c.metadata.get("had_clarification", False)
    )
    return clar / len(convs)


def mean_judge_scores(
    conversations: Iterable[Conversation],
) -> dict[str, float]:
    """Mean of each judge dimension across the corpus."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for conv in conversations:
        scores = conv.judge_scores
        if not scores:
            continue
        for key in (
            "tool_correctness",
            "grounding_fidelity",
            "naturalness",
            "task_completion",
            "mean",
        ):
            if key in scores and isinstance(scores[key], (int, float)):
                sums[key] = sums.get(key, 0.0) + float(scores[key])
                counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}

