"""
Builds a typed MultiDiGraph over endpoints capturing four kinds of
relationships used by the chain sampler.

Edge types
----------
SAME_TOOL
    Two endpoints belong to the same ToolBench tool. Strong signal:
    if a user searched, they are likely to also fetch details or act.
    Symmetric (added in both directions).

SAME_CATEGORY
    Two endpoints share a category. Weak signal: same domain, but not
    necessarily composable. Symmetric. Used as a fallback when no
    stronger edge exists.

PARAM_OVERLAP
    Two endpoints have at least one parameter with the same name.
    Medium signal: they handle similar entities (e.g. both take a
    `city`). Symmetric.

OUTPUT_TO_INPUT
    The strongest and most useful edge. A directed edge from A to B
    exists when some response field of A matches a required parameter
    of B under our conservative matching rule. This is what enables
    the sampler to find chains where IDs flow forward — the property
    that makes generated conversations look like real workflows.

Response-field enrichment
-------------------------
The loader leaves `response_fields` empty when ToolBench does not
declare them (which is most of the time). Since OUTPUT_TO_INPUT
detection depends on response fields, we run a heuristic enrichment
pass before building edges: endpoints whose name contains a list-like
verb (search, list, find, etc.) get synthetic `id` and `name` fields
injected. This is an explicit, documented approximation. At scale an
LLM-based annotation pass would give much better recall; the name
heuristic is chosen for determinism and cost during the exercise.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable, Literal

import networkx as nx

from convgen.registry.models import Endpoint, Registry, ResponseField

logger = logging.getLogger(__name__)

EdgeType = Literal[
    "SAME_TOOL",
    "SAME_CATEGORY",
    "PARAM_OVERLAP",
    "OUTPUT_TO_INPUT",
]

ALL_EDGE_TYPES: tuple[EdgeType, ...] = (
    "SAME_TOOL",
    "SAME_CATEGORY",
    "PARAM_OVERLAP",
    "OUTPUT_TO_INPUT",
)

# Default sampler weights per edge type. The sampler will use these to
# bias random walks toward stronger relationships. Exposed here so the
# graph and sampler agree on the vocabulary.
DEFAULT_EDGE_WEIGHTS: dict[EdgeType, float] = {
    "OUTPUT_TO_INPUT": 4.0,
    "SAME_TOOL": 2.0,
    "PARAM_OVERLAP": 1.0,
    "SAME_CATEGORY": 0.5,
}

# Verb-like prefixes that do not carry entity meaning. The first
# non-verb token in an endpoint name is treated as the entity noun.
_ENDPOINT_VERB_TOKENS: frozenset[str] = frozenset(
    {
        "get",
        "find",
        "search",
        "list",
        "fetch",
        "retrieve",
        "lookup",
        "browse",
        "all",
        "query",
        "default",
        "home",
        "random",
        "make",
        "do",
        "perform",
        "the",
        "a",
        "an",
    }
)


# --------------------------------------------------------------------------
# Response-field enrichment
# --------------------------------------------------------------------------


def _entity_from_endpoint(endpoint: Endpoint) -> str | None:
    """Extract a likely entity noun from the endpoint name.

    The rule is intentionally simple: split on underscores, skip
    leading verb-like tokens, then take the first remaining token.
    Applies light singularization to reduce noise in generated field
    names (e.g., "movies" -> "movie").
    """
    tokens = [t for t in endpoint.name.lower().split("_") if t]
    if not tokens:
        return None
    i = 0
    while i < len(tokens) and tokens[i] in _ENDPOINT_VERB_TOKENS:
        i += 1
    if i >= len(tokens):
        return None
    entity = tokens[i]
    if entity.endswith("ies") and len(entity) > 4:
        entity = entity[:-3] + "y"
    elif (
        entity.endswith("s")
        and len(entity) > 3
        and not entity.endswith("ss")
        and not entity.endswith("ws")
    ):
        entity = entity[:-1]
    return entity


def enrich_response_fields(endpoints: list[Endpoint]) -> list[Endpoint]:
    """
    Return a new list of endpoints with synthetic response fields for
    those that have none. Endpoints that already declare response fields
    are passed through untouched.

    Field names embed an entity noun from the endpoint name (e.g.
    `genre_id`, `trader_name`) so grounding can align on semantic slots
    rather than generic `id` / `name` keys. Count-like endpoints get an
    integer `count` field.
    """
    out: list[Endpoint] = []
    for ep in endpoints:
        if ep.response_fields:
            out.append(ep)
            continue

        entity = _entity_from_endpoint(ep)
        name_tokens = ep.name.lower().split("_")
        is_count = "count" in name_tokens or "number" in name_tokens

        fields: list[ResponseField] = []
        if is_count:
            fields.append(ResponseField(name="count", type="integer"))
            if entity:
                fields.append(
                    ResponseField(name=f"{entity}_category", type="string")
                )
            else:
                fields.append(ResponseField(name="category", type="string"))
        elif entity:
            fields.append(ResponseField(name=f"{entity}_id", type="string"))
            fields.append(ResponseField(name=f"{entity}_name", type="string"))
        else:
            fields.append(ResponseField(name="id", type="string"))
            fields.append(ResponseField(name="name", type="string"))

        enriched = ep.model_copy(
            update={"response_fields": fields}
        )
        out.append(enriched)
    return out


# --------------------------------------------------------------------------
# Field-matching rule for OUTPUT_TO_INPUT edges
# --------------------------------------------------------------------------


def _normalize(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


# Generic field names that are too common to support cross-tool matching.
# If both sides have a bare "id" (or "name"), they likely refer to different
# domain entities and should only match within the same tool. A response field
# named "hotel_id" is specific enough to match across tools.
_GENERIC_FIELD_NAMES: frozenset[str] = frozenset(
    {"id", "name", "type", "status", "value", "key", "title", "code"}
)


def _fields_match(
    response_field_name: str,
    param_name: str,
    from_tool: str,
    to_tool: str,
) -> bool:
    """
    Conservative match between a response field and a required parameter.

    Rules:
      1. Exact match on normalized names — BUT if the name is in the
         generic denylist, only allow the match within the same tool.
         This prevents chains where unrelated tools share a bare ``id``.
      2. Bare ``id`` on the producer side matches ``*_id`` params on the
         consumer side only within the same tool.

    Everything else is rejected. This favors precision over recall.
    """
    f = _normalize(response_field_name)
    p = _normalize(param_name)

    if f == p:
        if f in _GENERIC_FIELD_NAMES and from_tool != to_tool:
            return False
        return True

    if f == "id" and from_tool == to_tool and p.endswith("id") and p != "id":
        return True

    return False


# --------------------------------------------------------------------------
# ToolGraph — the facade the rest of the pipeline consumes
# --------------------------------------------------------------------------


class ToolGraph:
    """
    A typed MultiDiGraph over endpoints plus the Registry it was built from.

    Attributes
    ----------
    graph : networkx.MultiDiGraph
        Nodes are endpoint IDs. Each node has an `endpoint` attribute
        holding the full Endpoint object. Each edge has a `type`
        attribute drawn from ALL_EDGE_TYPES.
    registry : Registry
        The registry the graph was built from. Kept alongside the graph
        so downstream code can look up endpoints without a separate
        registry reference.
    """

    def __init__(self, graph: nx.MultiDiGraph, registry: Registry) -> None:
        self.graph = graph
        self.registry = registry

    # -- introspection --------------------------------------------------

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def endpoint_ids(self) -> list[str]:
        return list(self.graph.nodes)

    def get_endpoint(self, endpoint_id: str) -> Endpoint | None:
        data = self.graph.nodes.get(endpoint_id)
        if data is None:
            return None
        return data.get("endpoint")

    def neighbors(
        self,
        endpoint_id: str,
        edge_types: Iterable[EdgeType] | None = None,
    ) -> list[tuple[str, EdgeType]]:
        """
        Return (neighbor_id, edge_type) pairs for outgoing edges from the
        given endpoint. If `edge_types` is provided, only edges of those
        types are returned. Each parallel edge yields a separate pair.
        """
        if endpoint_id not in self.graph:
            return []
        wanted = set(edge_types) if edge_types is not None else None

        out: list[tuple[str, EdgeType]] = []
        for _, neighbor, data in self.graph.out_edges(endpoint_id, data=True):
            et: EdgeType = data["type"]
            if wanted is None or et in wanted:
                out.append((neighbor, et))
        return out

    def edge_type_counts(self) -> dict[EdgeType, int]:
        counts: dict[EdgeType, int] = {et: 0 for et in ALL_EDGE_TYPES}
        for _, _, data in self.graph.edges(data=True):
            counts[data["type"]] += 1
        return counts

    # -- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"graph": self.graph, "registry": self.registry}, f)
        logger.info(
            "Saved ToolGraph to %s (%d nodes, %d edges)",
            path,
            len(self),
            self.num_edges,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ToolGraph":
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls(graph=data["graph"], registry=data["registry"])


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def build_tool_graph(registry: Registry) -> ToolGraph:
    """
    Construct a ToolGraph from a Registry.

    Steps:
      1. Run response-field enrichment on the registry's endpoints and
         produce an enriched Registry (the original is left untouched).
      2. Create a MultiDiGraph with one node per endpoint.
      3. Add SAME_TOOL edges (symmetric).
      4. Add SAME_CATEGORY edges (symmetric).
      5. Add PARAM_OVERLAP edges (symmetric).
      6. Add OUTPUT_TO_INPUT edges (directed) using the enriched fields.

    The enriched Registry is what the ToolGraph stores, so downstream
    consumers see synthesized response fields as first-class data.
    """
    enriched_tools = []
    for tool in registry.tools:
        new_endpoints = enrich_response_fields(tool.endpoints)
        enriched_tools.append(tool.model_copy(update={"endpoints": new_endpoints}))
    enriched_registry = Registry(tools=enriched_tools)
    endpoints = enriched_registry.endpoints

    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    for ep in endpoints:
        graph.add_node(ep.id, endpoint=ep)

    # Pre-index for cheap lookups during edge construction.
    by_tool: dict[str, list[Endpoint]] = {}
    by_category: dict[str, list[Endpoint]] = {}
    for ep in endpoints:
        by_tool.setdefault(ep.tool_name, []).append(ep)
        by_category.setdefault(ep.category, []).append(ep)

    # 1. SAME_TOOL (symmetric, no self-loops)
    for tool_eps in by_tool.values():
        for a in tool_eps:
            for b in tool_eps:
                if a.id != b.id:
                    graph.add_edge(a.id, b.id, type="SAME_TOOL")

    # 2. SAME_CATEGORY (symmetric, skip same-tool pairs to avoid double-
    #    counting with SAME_TOOL — SAME_TOOL strictly dominates)
    for cat_eps in by_category.values():
        for a in cat_eps:
            for b in cat_eps:
                if a.id == b.id:
                    continue
                if a.tool_name == b.tool_name:
                    continue
                graph.add_edge(a.id, b.id, type="SAME_CATEGORY")

    # 3. PARAM_OVERLAP (symmetric, cross-tool only — same-tool pairs
    #    already covered by SAME_TOOL)
    param_index: dict[str, list[Endpoint]] = {}
    for ep in endpoints:
        for p in ep.parameters:
            param_index.setdefault(p.name.lower(), []).append(ep)
    for eps_sharing in param_index.values():
        for a in eps_sharing:
            for b in eps_sharing:
                if a.id == b.id or a.tool_name == b.tool_name:
                    continue
                graph.add_edge(a.id, b.id, type="PARAM_OVERLAP")

    # 4. OUTPUT_TO_INPUT (directed, conservative matching)
    for producer in endpoints:
        if not producer.response_fields:
            continue
        for consumer in endpoints:
            if producer.id == consumer.id:
                continue
            if not consumer.required_parameters:
                continue
            matched = False
            for rf in producer.response_fields:
                for rp in consumer.required_parameters:
                    if _fields_match(
                        rf.name,
                        rp.name,
                        from_tool=producer.tool_name,
                        to_tool=consumer.tool_name,
                    ):
                        matched = True
                        break
                if matched:
                    break
            if matched:
                graph.add_edge(producer.id, consumer.id, type="OUTPUT_TO_INPUT")

    tg = ToolGraph(graph=graph, registry=enriched_registry)
    logger.info(
        "Built ToolGraph: %d nodes, %d edges (by type: %s)",
        len(tg),
        tg.num_edges,
        tg.edge_type_counts(),
    )
    return tg
