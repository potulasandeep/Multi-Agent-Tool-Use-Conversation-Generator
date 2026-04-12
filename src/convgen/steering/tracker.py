"""
Cross-conversation coverage tracking and steering.

This module provides `CoverageTracker`, a concrete implementation of the
`BiasProvider` protocol from `convgen.graph.sampler`. The chain sampler uses
`bias(endpoint_id)` to down-weight endpoints that have already appeared in
earlier conversations during the same generation run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from convgen.graph.sampler import BiasProvider, SampledChain


@dataclass
class CoverageTracker(BiasProvider):
    """Counter-backed coverage tracker and BiasProvider.

    The bias formula is deterministic inverse-frequency weighting:
      bias(endpoint_id) = 1 / (1 + alpha * count)

    We also record tool, category, and adjacent-pair counts so the diversity
    experiment in a later step can be computed without replaying generation
    history from disk.
    """

    alpha: float = 1.0
    num_conversations: int = 0
    endpoint_counts: Counter[str] = field(default_factory=Counter)
    tool_counts: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    pair_counts: Counter[tuple[str, str]] = field(default_factory=Counter)

    def record(self, chain: SampledChain) -> None:
        """Record counts for a freshly sampled chain.

        Called once per conversation. Counts reflect what the sampler attempted
        to generate, regardless of whether the conversation eventually passed the
        judge.
        """
        if chain.length == 0:
            return

        self.num_conversations += 1
        for ep in chain.endpoints:
            self.endpoint_counts[ep.id] += 1
            self.tool_counts[ep.tool_name] += 1
            self.category_counts[ep.category] += 1

        for a, b in zip(chain.endpoints, chain.endpoints[1:]):
            self.pair_counts[(a.id, b.id)] += 1

    def bias(self, endpoint_id: str) -> float:
        """Return the steering weight for an endpoint ID.

        The value is always positive; endpoints with zero prior uses get weight
        1.0.
        """
        count = self.endpoint_counts.get(endpoint_id, 0)
        return 1.0 / (1.0 + self.alpha * count)

    @property
    def unique_endpoints(self) -> set[str]:
        return set(self.endpoint_counts.keys())

    @property
    def unique_tools(self) -> set[str]:
        return set(self.tool_counts.keys())

    @property
    def unique_categories(self) -> set[str]:
        return set(self.category_counts.keys())

    @property
    def unique_pairs(self) -> set[tuple[str, str]]:
        return set(self.pair_counts.keys())

    def snapshot(self) -> dict:
        """Return a plain dict snapshot suitable for CLI serialization."""
        return {
            "alpha": self.alpha,
            "num_conversations": self.num_conversations,
            "unique_endpoints": len(self.endpoint_counts),
            "unique_tools": len(self.tool_counts),
            "unique_categories": len(self.category_counts),
            "unique_pairs": len(self.pair_counts),
            "endpoint_counts": dict(self.endpoint_counts),
            "tool_counts": dict(self.tool_counts),
            "category_counts": dict(self.category_counts),
            "pair_counts": {
                f"{a}->{b}": c for (a, b), c in self.pair_counts.items()
            },
        }

    def reset(self) -> None:
        """Clear all recorded coverage state."""
        self.num_conversations = 0
        self.endpoint_counts.clear()
        self.tool_counts.clear()
        self.category_counts.clear()
        self.pair_counts.clear()

