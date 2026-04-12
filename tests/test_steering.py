"""Tests for cross-conversation steering (CoverageTracker)."""

from __future__ import annotations

from convgen.graph.sampler import SampledChain
from convgen.registry.models import Endpoint, Parameter, ResponseField
from convgen.steering.tracker import CoverageTracker


def _ep(
    endpoint_id: str, *, tool: str, category: str
) -> Endpoint:
    name = endpoint_id.split(".", 1)[1] if "." in endpoint_id else endpoint_id
    return Endpoint(
        id=endpoint_id,
        tool_name=tool,
        name=name,
        category=category,
        parameters=[Parameter(name="x", type="string", required=True)],
        response_fields=[ResponseField(name="y", type="string")],
    )


def test_coverage_tracker_record_bias_and_snapshot() -> None:
    tracker = CoverageTracker(alpha=1.0)

    ep_a1 = _ep("toolA.a1", tool="toolA", category="Cat1")
    ep_b1 = _ep("toolB.b1", tool="toolB", category="Cat2")
    ep_a2 = _ep("toolA.a2", tool="toolA", category="Cat1")

    # Chain 1: A1 -> B1
    tracker.record(
        SampledChain(
            endpoints=[ep_a1, ep_b1],
            pattern="sequential",
        )
    )

    # Chain 2: A1 -> A2
    tracker.record(
        SampledChain(
            endpoints=[ep_a1, ep_a2],
            pattern="sequential",
        )
    )

    assert tracker.endpoint_counts[ep_a1.id] == 2
    assert tracker.endpoint_counts[ep_b1.id] == 1
    assert tracker.endpoint_counts[ep_a2.id] == 1

    # bias = 1 / (1 + alpha * count)
    assert tracker.bias(ep_a1.id) == 1.0 / (1.0 + 1.0 * 2.0)
    assert tracker.bias(ep_b1.id) == 1.0 / (1.0 + 1.0 * 1.0)
    assert tracker.bias("unknown.endpoint") == 1.0

    snap = tracker.snapshot()
    assert snap["alpha"] == 1.0
    assert snap["num_conversations"] == 2
    assert snap["endpoint_counts"][ep_a1.id] == 2
    assert snap["unique_endpoints"] == 3

    # Adjacent pairs:
    #  Chain1: (A1,B1)
    #  Chain2: (A1,A2)
    assert snap["pair_counts"][f"{ep_a1.id}->{ep_b1.id}"] == 1
    assert snap["pair_counts"][f"{ep_a1.id}->{ep_a2.id}"] == 1


def test_coverage_tracker_reset() -> None:
    tracker = CoverageTracker(alpha=1.0)
    ep_a = _ep("toolA.a", tool="toolA", category="Cat1")
    tracker.record(SampledChain(endpoints=[ep_a], pattern="sequential"))
    assert tracker.num_conversations == 1
    assert tracker.endpoint_counts[ep_a.id] == 1

    tracker.reset()
    assert tracker.num_conversations == 0
    assert tracker.endpoint_counts == {}
    assert tracker.tool_counts == {}
    assert tracker.category_counts == {}
    assert tracker.pair_counts == {}

