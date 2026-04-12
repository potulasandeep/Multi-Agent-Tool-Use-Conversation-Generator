"""
Tool-chain sampler over a ToolGraph.

This module turns the structural graph from `graph.builder` into actual
chains that the conversation generator will consume. It exposes:

- `SamplingConstraints`: declarative knobs (length, category filter,
  pattern, etc.) callers use to describe what they want.
- `SampledChain`: the result, containing the ordered endpoints and the
  pattern used.
- `BiasProvider`: a Protocol that the cross-conversation steering layer
  (implemented in Step 9) will satisfy. The sampler multiplies candidate
  weights by `bias(endpoint_id)` when a provider is supplied.
- `RandomWalkSampler`: the basic graph walker. One attempt per call.
- `ConstrainedSampler`: wraps the walker with retry logic for hard
  constraints.

Hard requirement from the assignment: the conversation generator must
obtain its chains from this module, never from a hardcoded list. The
two-class structure (walker + constrained wrapper) keeps responsibilities
clean: the walker is pure and trivially testable; the constrained
sampler is retry policy on top.

Reproducibility: every sampler accepts an injected `random.Random`
instance. Two samplers constructed with `random.Random(seed)` and the
same graph will produce identical chains. This is what makes the
diversity experiment in Step 12 reproducible.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Literal, Protocol

from convgen.graph.builder import (
    DEFAULT_EDGE_WEIGHTS,
    EdgeType,
    ToolGraph,
)
from convgen.registry.models import Endpoint

logger = logging.getLogger(__name__)


ChainPattern = Literal["sequential", "parallel"]


# --------------------------------------------------------------------------
# Cross-conversation steering hook
# --------------------------------------------------------------------------


class BiasProvider(Protocol):
    """Protocol for cross-conversation steering.

    The sampler multiplies candidate weights by `bias(endpoint_id)`.
    Higher return values make an endpoint more likely to be selected.
    Implementations live in `convgen.steering` (Step 9).

    The interface is deliberately narrow — only an endpoint ID — so it
    is trivial to satisfy from any backend (in-memory counter, vector
    store, learned policy). The cost is that the bias function cannot
    see graph context or chain-so-far; document this as a limitation.
    """

    def bias(self, endpoint_id: str) -> float: ...


# --------------------------------------------------------------------------
# Public dataclasses
# --------------------------------------------------------------------------


@dataclass
class SamplingConstraints:
    """Declarative knobs for chain sampling.

    All fields are optional. If `length` is set, it overrides the
    `min_length`/`max_length` range. `must_include_*` constraints are
    enforced by the ConstrainedSampler via retries.

    When ``require_grounded_anchor`` is True (default), at least one
    endpoint in the chain must declare a required parameter so the user
    turn can anchor concrete argument values.
    """

    length: int | None = None
    min_length: int = 2
    max_length: int = 5
    must_include_category: str | None = None
    must_include_tool: str | None = None
    pattern: ChainPattern = "sequential"
    prefer_edge_types: list[EdgeType] = field(
        default_factory=lambda: ["OUTPUT_TO_INPUT", "SAME_TOOL"]
    )
    max_retries: int = 20
    min_distinct_tools: int = 1
    """Minimum number of distinct tools the chain must include."""
    # Reject all-parameterless chains (no user-text anchor for required args).
    require_grounded_anchor: bool = True

    def target_length(self, rng: random.Random) -> int:
        if self.length is not None:
            return self.length
        return rng.randint(self.min_length, self.max_length)


def cli_generation_slot_constraints(
    i: int,
    n: int,
    *,
    min_length: int,
    max_length: int,
) -> SamplingConstraints:
    """Constraints for conversation index *i* in a run of *n* conversations.

    Mirrors the deterministic 55/15/30 split in ``convgen generate`` so
    e2e tests and the CLI agree on corpus shape (multi-tool + parallel
    fractions) without duplicating the split logic in two places.
    """
    position = i / n if n > 0 else 0.0
    if position < 0.55:
        pattern: ChainPattern = "sequential"
        min_distinct_tools = 2
    elif position < 0.70:
        pattern = "parallel"
        min_distinct_tools = 2
    else:
        pattern = "sequential"
        min_distinct_tools = 1
    max_retries = 50 if min_distinct_tools >= 2 else 20
    return SamplingConstraints(
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        min_distinct_tools=min_distinct_tools,
        max_retries=max_retries,
    )


@dataclass
class SampledChain:
    """Output of a sampler call. A flat ordered list of endpoints
    plus the pattern that produced them."""

    endpoints: list[Endpoint]
    pattern: ChainPattern

    @property
    def length(self) -> int:
        return len(self.endpoints)

    @property
    def endpoint_ids(self) -> list[str]:
        return [ep.id for ep in self.endpoints]

    @property
    def tools_used(self) -> set[str]:
        return {ep.tool_name for ep in self.endpoints}

    @property
    def categories_used(self) -> set[str]:
        return {ep.category for ep in self.endpoints}


class SamplerError(Exception):
    """Raised when ConstrainedSampler exhausts retries without satisfying
    its hard constraints."""


# --------------------------------------------------------------------------
# RandomWalkSampler — the basic walker
# --------------------------------------------------------------------------


class RandomWalkSampler:
    """Walks the tool graph following weighted edges to produce a chain.

    One attempt per `sample()` call. Dead-end walks are returned as-is
    (possibly shorter than the target length). Retry/repair policy
    belongs to ConstrainedSampler, not here.
    """

    def __init__(
        self,
        graph: ToolGraph,
        rng: random.Random | None = None,
        bias_provider: BiasProvider | None = None,
    ) -> None:
        if len(graph) == 0:
            raise ValueError("Cannot sample from an empty ToolGraph")
        self.graph = graph
        self.rng = rng or random.Random()
        self.bias = bias_provider

    # -- start node selection ------------------------------------------

    def _candidate_start_nodes(
        self, constraints: SamplingConstraints
    ) -> list[str]:
        """Filter starting nodes by must_include constraints, if any.

        Note: filtering at the start gives the constraint a much better
        chance of being satisfied without retries. The ConstrainedSampler
        still verifies after the walk.
        """
        all_ids = self.graph.endpoint_ids()
        if constraints.must_include_tool:
            return [
                endpoint_id
                for endpoint_id in all_ids
                if self.graph.get_endpoint(endpoint_id).tool_name
                == constraints.must_include_tool
            ]
        if constraints.must_include_category:
            return [
                endpoint_id
                for endpoint_id in all_ids
                if self.graph.get_endpoint(endpoint_id).category
                == constraints.must_include_category
            ]
        return all_ids

    def _pick_start(self, candidates: list[str]) -> str:
        if self.bias is None:
            return self.rng.choice(candidates)
        weights = [max(self.bias.bias(endpoint_id), 1e-6) for endpoint_id in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    # -- weighted neighbor selection -----------------------------------

    def _weighted_choice(
        self,
        candidates: list[tuple[str, EdgeType]],
        visited: set[str],
    ) -> str | None:
        """Pick one neighbor weighted by edge type and optional bias.

        Returns None if all candidates are already visited or no
        candidate has positive weight.
        """
        weights: list[float] = []
        valid: list[str] = []
        for neighbor_id, edge_type in candidates:
            if neighbor_id in visited:
                continue
            weight = DEFAULT_EDGE_WEIGHTS[edge_type]
            if self.bias is not None:
                weight *= max(self.bias.bias(neighbor_id), 0.0)
            if weight > 0:
                weights.append(weight)
                valid.append(neighbor_id)
        if not valid:
            return None
        return self.rng.choices(valid, weights=weights, k=1)[0]

    # -- pattern walks --------------------------------------------------

    def _walk_sequential(
        self,
        start_id: str,
        target_length: int,
        prefer_edge_types: list[EdgeType],
    ) -> list[Endpoint]:
        """Walk the graph following preferred edge types first, falling
        back to any outgoing edge if no preferred neighbor is available."""
        chain_ids: list[str] = [start_id]
        visited: set[str] = {start_id}

        while len(chain_ids) < target_length:
            current = chain_ids[-1]
            preferred = self.graph.neighbors(current, edge_types=prefer_edge_types)
            chosen = self._weighted_choice(preferred, visited)
            if chosen is None:
                fallback = self.graph.neighbors(current)
                chosen = self._weighted_choice(fallback, visited)
            if chosen is None:
                break  # dead end
            chain_ids.append(chosen)
            visited.add(chosen)

        return [self.graph.get_endpoint(endpoint_id) for endpoint_id in chain_ids]

    def _walk_parallel(
        self,
        start_id: str,
        target_length: int,
    ) -> list[Endpoint]:
        """Hub-and-spoke chain: start endpoint plus siblings sharing a
        parameter (PARAM_OVERLAP) or category (SAME_CATEGORY)."""
        chain_ids: list[str] = [start_id]
        visited: set[str] = {start_id}

        siblings_raw = self.graph.neighbors(
            start_id, edge_types=["PARAM_OVERLAP", "SAME_CATEGORY"]
        )
        # Deduplicate parallel edges to the same neighbor.
        seen: set[str] = set()
        siblings: list[tuple[str, EdgeType]] = []
        for neighbor_id, edge_type in siblings_raw:
            if neighbor_id not in seen:
                seen.add(neighbor_id)
                siblings.append((neighbor_id, edge_type))

        while len(chain_ids) < target_length and siblings:
            chosen = self._weighted_choice(siblings, visited)
            if chosen is None:
                break
            chain_ids.append(chosen)
            visited.add(chosen)
            siblings = [
                (neighbor_id, edge_type)
                for neighbor_id, edge_type in siblings
                if neighbor_id != chosen
            ]

        return [self.graph.get_endpoint(endpoint_id) for endpoint_id in chain_ids]

    # -- public API -----------------------------------------------------

    def sample(self, constraints: SamplingConstraints) -> SampledChain:
        """One attempt to produce a chain. Does not retry on constraint
        failure — that's ConstrainedSampler's job."""
        starts = self._candidate_start_nodes(constraints)
        if not starts:
            raise SamplerError(
                f"No starting nodes match constraints: {constraints}"
            )
        target_length = constraints.target_length(self.rng)
        start_id = self._pick_start(starts)

        if constraints.pattern == "sequential":
            endpoints = self._walk_sequential(
                start_id, target_length, constraints.prefer_edge_types
            )
        elif constraints.pattern == "parallel":
            endpoints = self._walk_parallel(start_id, target_length)
        else:
            raise ValueError(f"Unknown pattern: {constraints.pattern}")

        return SampledChain(endpoints=endpoints, pattern=constraints.pattern)


# --------------------------------------------------------------------------
# ConstrainedSampler — retry wrapper enforcing hard constraints
# --------------------------------------------------------------------------


class ConstrainedSampler:
    """Wraps RandomWalkSampler with retry logic for hard constraints.

    On each `sample()` call, retries the underlying walker up to
    `constraints.max_retries` times until a chain satisfies all
    constraints (exact length, must_include_*). Raises SamplerError
    if no attempt succeeds.
    """

    def __init__(
        self,
        graph: ToolGraph,
        rng: random.Random | None = None,
        bias_provider: BiasProvider | None = None,
    ) -> None:
        self.walker = RandomWalkSampler(
            graph, rng=rng, bias_provider=bias_provider
        )
        self.rng = self.walker.rng

    def _satisfies(
        self, chain: SampledChain, constraints: SamplingConstraints
    ) -> bool:
        if constraints.length is not None:
            if chain.length != constraints.length:
                return False
        else:
            if not (
                constraints.min_length
                <= chain.length
                <= constraints.max_length
            ):
                return False
        if (
            constraints.must_include_category
            and constraints.must_include_category not in chain.categories_used
        ):
            return False
        if (
            constraints.must_include_tool
            and constraints.must_include_tool not in chain.tools_used
        ):
            return False
        if len(chain.tools_used) < constraints.min_distinct_tools:
            return False
        if constraints.require_grounded_anchor:
            has_anchor = any(
                len(ep.required_parameters) > 0 for ep in chain.endpoints
            )
            if not has_anchor:
                return False
        return True

    def sample(self, constraints: SamplingConstraints) -> SampledChain:
        last_chain: SampledChain | None = None
        for attempt in range(constraints.max_retries):
            chain = self.walker.sample(constraints)
            if self._satisfies(chain, constraints):
                logger.debug(
                    "ConstrainedSampler succeeded on attempt %d", attempt + 1
                )
                return chain
            last_chain = chain
        raise SamplerError(
            f"Could not satisfy constraints {constraints} in "
            f"{constraints.max_retries} attempts. Last chain: "
            f"{last_chain.endpoint_ids if last_chain else None}"
        )
