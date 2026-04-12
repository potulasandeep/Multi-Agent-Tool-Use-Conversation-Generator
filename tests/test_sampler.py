"""Tests for the chain sampler.

Each test builds its graph in-line via small helpers so the test body
shows exactly which structural conditions are being exercised.
"""

from __future__ import annotations

import random

import pytest

from convgen.graph.builder import ToolGraph, build_tool_graph
from convgen.graph.sampler import (
    BiasProvider,
    ConstrainedSampler,
    RandomWalkSampler,
    SampledChain,
    SamplerError,
    SamplingConstraints,
    cli_generation_slot_constraints,
)
from convgen.registry.models import (
    Endpoint,
    Parameter,
    Registry,
    ResponseField,
    Tool,
)


# ---------------------- fixture helpers ----------------------


def _ep(
    tool: str,
    name: str,
    category: str = "Travel",
    required: list[tuple[str, str]] | None = None,
    optional: list[tuple[str, str]] | None = None,
    response: list[tuple[str, str]] | None = None,
) -> Endpoint:
    return Endpoint(
        id=f"{tool}.{name}",
        tool_name=tool,
        name=name,
        category=category,
        parameters=[
            Parameter(name=n, type=t, required=True) for n, t in (required or [])
        ]
        + [
            Parameter(name=n, type=t, required=False) for n, t in (optional or [])
        ],
        response_fields=[ResponseField(name=n, type=t) for n, t in (response or [])],
    )


def _registry(*endpoints: Endpoint) -> Registry:
    by_tool: dict[str, list[Endpoint]] = {}
    cat_by_tool: dict[str, str] = {}
    for ep in endpoints:
        by_tool.setdefault(ep.tool_name, []).append(ep)
        cat_by_tool[ep.tool_name] = ep.category
    tools = [
        Tool(name=tool_name, category=cat_by_tool[tool_name], endpoints=eps)
        for tool_name, eps in by_tool.items()
    ]
    return Registry(tools=tools)


@pytest.fixture
def small_graph() -> ToolGraph:
    """A modest cross-domain graph used by most sampler tests."""
    reg = _registry(
        # Travel — hotels (clear OUTPUT_TO_INPUT chain)
        _ep(
            "hotels",
            "search",
            category="Travel",
            required=[("city", "string")],
            response=[("id", "string"), ("name", "string")],
        ),
        _ep(
            "hotels",
            "get_details",
            category="Travel",
            required=[("hotel_id", "string")],
        ),
        _ep(
            "hotels",
            "book",
            category="Travel",
            required=[("hotel_id", "string"), ("check_in", "string")],
        ),
        # Travel — flights
        _ep(
            "flights",
            "search",
            category="Travel",
            required=[("from_city", "string"), ("to_city", "string")],
            response=[("id", "string")],
        ),
        _ep(
            "flights",
            "book",
            category="Travel",
            required=[("flight_id", "string")],
        ),
        # Weather (shares `city` with hotels.search)
        _ep(
            "weather",
            "forecast",
            category="Weather",
            required=[("city", "string"), ("date", "string")],
        ),
        # Finance (isolated)
        _ep(
            "stocks",
            "quote",
            category="Finance",
            required=[("symbol", "string")],
        ),
    )
    return build_tool_graph(reg)


# ---------------------- SamplingConstraints ----------------------


class TestSamplingConstraints:
    def test_target_length_exact(self):
        constraints = SamplingConstraints(length=4)
        assert constraints.target_length(random.Random(0)) == 4

    def test_target_length_within_range(self):
        constraints = SamplingConstraints(min_length=2, max_length=5)
        rng = random.Random(0)
        for _ in range(50):
            assert 2 <= constraints.target_length(rng) <= 5


class TestCliGenerationSlotConstraints:
    def test_55_15_30_split_counts(self) -> None:
        n = 100
        seq_multi = parallel_multi = single = 0
        for i in range(n):
            c = cli_generation_slot_constraints(i, n, min_length=2, max_length=5)
            if c.min_distinct_tools >= 2 and c.pattern == "sequential":
                seq_multi += 1
            elif c.min_distinct_tools >= 2 and c.pattern == "parallel":
                parallel_multi += 1
            else:
                single += 1
        assert seq_multi == 55
        assert parallel_multi == 15
        assert single == 30


# ---------------------- RandomWalkSampler ----------------------


class TestRandomWalkSampler:
    def test_empty_graph_rejected(self):
        empty = build_tool_graph(Registry())
        with pytest.raises(ValueError):
            RandomWalkSampler(empty)

    def test_returns_sampled_chain(self, small_graph: ToolGraph):
        sampler = RandomWalkSampler(small_graph, rng=random.Random(42))
        chain = sampler.sample(SamplingConstraints(length=3))
        assert isinstance(chain, SampledChain)
        assert chain.length >= 1
        assert all(isinstance(endpoint, Endpoint) for endpoint in chain.endpoints)

    def test_reproducible_with_same_seed(self, small_graph: ToolGraph):
        sampler_one = RandomWalkSampler(small_graph, rng=random.Random(123))
        sampler_two = RandomWalkSampler(small_graph, rng=random.Random(123))
        chain_one = sampler_one.sample(SamplingConstraints(length=3))
        chain_two = sampler_two.sample(SamplingConstraints(length=3))
        assert chain_one.endpoint_ids == chain_two.endpoint_ids

    def test_different_seeds_can_produce_different_chains(self, small_graph: ToolGraph):
        # Not strictly guaranteed for any single seed pair, but with 10
        # seeds we should see at least two distinct chains.
        chains = set()
        for seed in range(10):
            sampler = RandomWalkSampler(small_graph, rng=random.Random(seed))
            chain = sampler.sample(SamplingConstraints(length=3))
            chains.add(tuple(chain.endpoint_ids))
        assert len(chains) > 1

    def test_no_revisits(self, small_graph: ToolGraph):
        for seed in range(20):
            sampler = RandomWalkSampler(small_graph, rng=random.Random(seed))
            chain = sampler.sample(SamplingConstraints(length=4))
            assert len(chain.endpoint_ids) == len(set(chain.endpoint_ids))

    def test_dead_end_returns_short_chain(self, small_graph: ToolGraph):
        # The Finance category contains exactly one endpoint, so any walk
        # starting there cannot grow under same-category constraints.
        sampler = RandomWalkSampler(small_graph, rng=random.Random(0))
        chain = sampler.sample(
            SamplingConstraints(length=5, must_include_category="Finance")
        )
        # The walker can fall back to other edge types, but the start
        # node will be `stocks.quote`. We just assert no crash and some
        # length ≤ 5.
        assert 1 <= chain.length <= 5

    def test_sequential_chain_prefers_output_to_input(self, small_graph: ToolGraph):
        # Run many samples; the dominant 3-step chain should follow the
        # hotels OUTPUT_TO_INPUT path more often than not.
        sampler = RandomWalkSampler(small_graph, rng=random.Random(0))
        hits = 0
        for _ in range(50):
            chain = sampler.sample(
                SamplingConstraints(length=3, must_include_tool="hotels")
            )
            if (
                chain.endpoint_ids[0] == "hotels.search"
                and "hotels.book" in chain.endpoint_ids
            ):
                hits += 1
        assert hits >= 10  # not strict, just sanity that the bias works


class TestParallelPattern:
    def test_parallel_returns_pattern_label(self, small_graph: ToolGraph):
        sampler = RandomWalkSampler(small_graph, rng=random.Random(0))
        chain = sampler.sample(SamplingConstraints(length=3, pattern="parallel"))
        assert chain.pattern == "parallel"
        assert chain.length >= 1

    def test_parallel_with_unknown_pattern_raises(self, small_graph: ToolGraph):
        sampler = RandomWalkSampler(small_graph, rng=random.Random(0))
        bad = SamplingConstraints(length=2)
        bad.pattern = "wibble"  # type: ignore[assignment]
        with pytest.raises(ValueError):
            sampler.sample(bad)


# ---------------------- ConstrainedSampler ----------------------


class TestConstrainedSampler:
    def test_exact_length_satisfied(self, small_graph: ToolGraph):
        sampler = ConstrainedSampler(small_graph, rng=random.Random(42))
        chain = sampler.sample(SamplingConstraints(length=3))
        assert chain.length == 3

    def test_must_include_category(self, small_graph: ToolGraph):
        sampler = ConstrainedSampler(small_graph, rng=random.Random(42))
        chain = sampler.sample(
            SamplingConstraints(length=2, must_include_category="Travel")
        )
        assert "Travel" in chain.categories_used

    def test_must_include_tool(self, small_graph: ToolGraph):
        sampler = ConstrainedSampler(small_graph, rng=random.Random(42))
        chain = sampler.sample(
            SamplingConstraints(length=2, must_include_tool="hotels")
        )
        assert "hotels" in chain.tools_used

    def test_unsatisfiable_constraint_raises(self, small_graph: ToolGraph):
        sampler = ConstrainedSampler(small_graph, rng=random.Random(42))
        with pytest.raises(SamplerError):
            sampler.sample(
                SamplingConstraints(
                    length=10,
                    must_include_tool="nonexistent_tool",
                    max_retries=3,
                )
            )

    def test_reproducible(self, small_graph: ToolGraph):
        sampler_one = ConstrainedSampler(small_graph, rng=random.Random(7))
        sampler_two = ConstrainedSampler(small_graph, rng=random.Random(7))
        chain_one = sampler_one.sample(SamplingConstraints(length=3))
        chain_two = sampler_two.sample(SamplingConstraints(length=3))
        assert chain_one.endpoint_ids == chain_two.endpoint_ids

    def test_min_distinct_tools_constraint(self, small_graph: ToolGraph):
        sampler = ConstrainedSampler(small_graph, rng=random.Random(42))
        chain = sampler.sample(
            SamplingConstraints(length=3, min_distinct_tools=2, max_retries=50)
        )
        assert len(chain.tools_used) >= 2

    def test_min_distinct_tools_unsatisfiable_raises(self):
        reg = _registry(
            _ep("solo", "search", required=[("city", "string")]),
            _ep("solo", "book", required=[("id", "string")]),
        )
        graph = build_tool_graph(reg)
        sampler = ConstrainedSampler(graph, rng=random.Random(0))
        with pytest.raises(SamplerError):
            sampler.sample(
                SamplingConstraints(
                    length=2,
                    min_distinct_tools=2,
                    max_retries=5,
                )
            )


# ---------------------- BiasProvider integration ----------------------


class _FixedBias:
    """Test double satisfying the BiasProvider Protocol."""

    def __init__(self, target: str, target_weight: float = 100.0):
        self.target = target
        self.target_weight = target_weight

    def bias(self, endpoint_id: str) -> float:
        return self.target_weight if endpoint_id == self.target else 0.01


class TestBiasProvider:
    def test_protocol_is_satisfied_structurally(self):
        # _FixedBias has no inheritance from BiasProvider but should
        # still satisfy the Protocol structurally.
        bias_provider: BiasProvider = _FixedBias("hotels.search")
        assert bias_provider.bias("hotels.search") == 100.0
        assert bias_provider.bias("anything_else") == 0.01

    def test_strong_bias_dominates_selection(self, small_graph: ToolGraph):
        target = "hotels.search"
        sampler = ConstrainedSampler(
            small_graph,
            rng=random.Random(0),
            bias_provider=_FixedBias(target),
        )
        appearances = 0
        for _ in range(20):
            chain = sampler.sample(SamplingConstraints(length=2))
            if target in chain.endpoint_ids:
                appearances += 1
        # With a 10000:1 weight ratio, the target should appear in nearly
        # every chain. We assert ≥75% as a robust threshold.
        assert appearances >= 15

    def test_no_bias_provider_uses_uniform(self, small_graph: ToolGraph):
        # Without a bias provider, sampling many chains should produce
        # several distinct starting nodes (i.e., we're not stuck on one).
        sampler = ConstrainedSampler(small_graph, rng=random.Random(0))
        starts = set()
        for _ in range(30):
            chain = sampler.sample(SamplingConstraints(length=2))
            starts.add(chain.endpoint_ids[0])
        assert len(starts) >= 3


class TestGroundedAnchor:
    def test_require_grounded_anchor_rejects_all_zero_param_chains(self):
        reg = _registry(
            _ep("a", "home"),
            _ep("b", "default"),
            _ep("a", "list"),
        )
        graph = build_tool_graph(reg)
        sampler = ConstrainedSampler(graph, rng=random.Random(0))
        with pytest.raises(SamplerError):
            sampler.sample(
                SamplingConstraints(
                    length=2,
                    require_grounded_anchor=True,
                    max_retries=5,
                )
            )

    def test_require_grounded_anchor_allows_mixed_chain(self):
        reg = _registry(
            _ep("a", "search", required=[("city", "string")]),
            _ep("a", "details"),
        )
        graph = build_tool_graph(reg)
        sampler = ConstrainedSampler(graph, rng=random.Random(0))
        chain = sampler.sample(
            SamplingConstraints(length=2, require_grounded_anchor=True)
        )
        assert chain.length == 2
        assert any(len(ep.required_parameters) > 0 for ep in chain.endpoints)
