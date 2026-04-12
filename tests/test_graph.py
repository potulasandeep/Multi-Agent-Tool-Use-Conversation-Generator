"""Tests for the tool graph builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from convgen.graph.builder import (
    ALL_EDGE_TYPES,
    DEFAULT_EDGE_WEIGHTS,
    ToolGraph,
    _entity_from_endpoint,
    _fields_match,
    build_tool_graph,
    enrich_response_fields,
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


# ---------------------- helper function tests ----------------------


class TestFieldsMatch:
    def test_exact_match(self):
        assert _fields_match("hotel_id", "hotel_id", "hotels", "hotels")

    def test_case_insensitive(self):
        assert _fields_match("HotelId", "hotel_id", "hotels", "hotels")

    def test_underscore_insensitive(self):
        assert _fields_match("hotel_id", "hotelid", "hotels", "hotels")

    def test_bare_id_matches_same_tool_star_id(self):
        assert _fields_match("id", "hotel_id", "hotels", "hotels")

    def test_bare_id_does_not_match_cross_tool(self):
        assert not _fields_match("id", "hotel_id", "flights", "hotels")

    def test_bare_id_does_not_match_bare_id_cross_tool(self):
        assert not _fields_match("id", "id", "x", "y")

    def test_bare_id_matches_bare_id_same_tool(self):
        assert _fields_match("id", "id", "hotels", "hotels")

    def test_generic_name_does_not_match_cross_tool(self):
        assert not _fields_match("name", "name", "anthill", "genius")

    def test_generic_name_matches_same_tool(self):
        assert _fields_match("status", "status", "api", "api")

    def test_unrelated_fields_dont_match(self):
        assert not _fields_match("price", "hotel_id", "hotels", "hotels")

    def test_partial_string_is_not_a_match(self):
        # `hotel` inside `hotel_id` must NOT match — we require full name
        # equality or the bare-id rule.
        assert not _fields_match("hotel", "hotel_id", "hotels", "hotels")


class TestEnrichResponseFields:
    def test_listy_endpoint_gets_entity_aware_fields(self):
        eps = [_ep("hotels", "search", required=[("city", "string")])]
        enriched = enrich_response_fields(eps)
        fields = {f.name for f in enriched[0].response_fields}
        # "search" contains no entity noun, so we fall back to bare keys.
        assert fields == {"id", "name"}

    def test_non_listy_endpoint_now_also_enriched(self):
        eps = [_ep("hotels", "cancel", required=[("booking_id", "string")])]
        enriched = enrich_response_fields(eps)
        assert len(enriched[0].response_fields) > 0

    def test_entity_extracted_from_get_by_pattern(self):
        eps = [_ep("movies", "get_genre_by_id", required=[("id", "string")])]
        enriched = enrich_response_fields(eps)
        names = {f.name for f in enriched[0].response_fields}
        assert "genre_id" in names
        assert "genre_name" in names

    def test_count_endpoint_gets_count_field(self):
        eps = [
            _ep(
                "airports",
                "airport_count_by_country",
                required=[("country", "string")],
            )
        ]
        enriched = enrich_response_fields(eps)
        names = {f.name for f in enriched[0].response_fields}
        assert "count" in names

    def test_existing_response_fields_not_overwritten(self):
        eps = [
            _ep(
                "hotels",
                "search",
                required=[("city", "string")],
                response=[("already", "string")],
            )
        ]
        enriched = enrich_response_fields(eps)
        assert [f.name for f in enriched[0].response_fields] == ["already"]

    @pytest.mark.parametrize(
        "name",
        ["search", "list_hotels", "find_flights", "get_user", "fetch_quote"],
    )
    def test_various_list_verbs_trigger_enrichment(self, name):
        eps = [_ep("t", name)]
        enriched = enrich_response_fields(eps)
        assert len(enriched[0].response_fields) == 2


class TestEntityFromEndpoint:
    def test_extracts_entity_examples(self):
        assert _entity_from_endpoint(_ep("x", "Get_genre_by_id")) == "genre"
        assert _entity_from_endpoint(_ep("x", "Search_Trader")) == "trader"
        assert _entity_from_endpoint(_ep("x", "Airport_Count_by_Use")) == "airport"
        assert _entity_from_endpoint(_ep("x", "news_detail")) == "news"
        assert _entity_from_endpoint(_ep("x", "random_idol")) == "idol"
        assert _entity_from_endpoint(_ep("x", "default")) is None


# ---------------------- full build pipeline tests ----------------------


class TestBuildToolGraph:
    def test_empty_registry(self):
        tg = build_tool_graph(Registry())
        assert len(tg) == 0
        assert tg.num_edges == 0

    def test_nodes_have_endpoint_attached(self):
        reg = _registry(_ep("hotels", "search"))
        tg = build_tool_graph(reg)
        ep = tg.get_endpoint("hotels.search")
        assert ep is not None
        assert ep.name == "search"

    def test_same_tool_edges(self):
        reg = _registry(
            _ep("hotels", "search"),
            _ep("hotels", "book"),
        )
        tg = build_tool_graph(reg)
        neighbors = tg.neighbors("hotels.search", edge_types=["SAME_TOOL"])
        assert ("hotels.book", "SAME_TOOL") in neighbors

    def test_same_category_edges_cross_tool_only(self):
        reg = _registry(
            _ep("hotels", "search", category="Travel"),
            _ep("flights", "search", category="Travel"),
            _ep("hotels", "book", category="Travel"),
        )
        tg = build_tool_graph(reg)
        # hotels.search ↔ flights.search should have SAME_CATEGORY
        assert ("flights.search", "SAME_CATEGORY") in tg.neighbors(
            "hotels.search", edge_types=["SAME_CATEGORY"]
        )
        # hotels.search ↔ hotels.book should NOT — they share a tool, so
        # SAME_TOOL supersedes SAME_CATEGORY.
        assert ("hotels.book", "SAME_CATEGORY") not in tg.neighbors(
            "hotels.search", edge_types=["SAME_CATEGORY"]
        )

    def test_param_overlap_edges_cross_tool(self):
        reg = _registry(
            _ep(
                "hotels",
                "search",
                category="Travel",
                required=[("city", "string")],
            ),
            _ep(
                "weather",
                "forecast",
                category="Weather",
                required=[("city", "string"), ("date", "string")],
            ),
        )
        tg = build_tool_graph(reg)
        neighbors = tg.neighbors("hotels.search", edge_types=["PARAM_OVERLAP"])
        assert ("weather.forecast", "PARAM_OVERLAP") in neighbors

    def test_output_to_input_edge_same_tool_via_bare_id(self):
        reg = _registry(
            _ep(
                "hotels",
                "search",
                required=[("city", "string")],
                response=[("id", "string"), ("name", "string")],
            ),
            _ep(
                "hotels",
                "book",
                required=[("hotel_id", "string"), ("check_in", "string")],
            ),
        )
        tg = build_tool_graph(reg)
        neighbors = tg.neighbors("hotels.search", edge_types=["OUTPUT_TO_INPUT"])
        assert ("hotels.book", "OUTPUT_TO_INPUT") in neighbors

    def test_output_to_input_edge_exact_match_cross_tool(self):
        reg = _registry(
            _ep(
                "users",
                "lookup",
                required=[("email", "string")],
                response=[("user_id", "string")],
            ),
            _ep(
                "orders",
                "list",
                required=[("user_id", "string")],
            ),
        )
        tg = build_tool_graph(reg)
        neighbors = tg.neighbors("users.lookup", edge_types=["OUTPUT_TO_INPUT"])
        assert ("orders.list", "OUTPUT_TO_INPUT") in neighbors

    def test_output_to_input_does_not_connect_bare_id_cross_tool(self):
        reg = _registry(
            _ep(
                "hotels",
                "search",
                required=[("city", "string")],
                response=[("id", "string")],
            ),
            _ep(
                "flights",
                "book",
                required=[("flight_id", "string")],
            ),
        )
        tg = build_tool_graph(reg)
        neighbors = tg.neighbors("hotels.search", edge_types=["OUTPUT_TO_INPUT"])
        # No OUTPUT_TO_INPUT edge should cross from hotels to flights via
        # bare `id`.
        assert ("flights.book", "OUTPUT_TO_INPUT") not in neighbors

    def test_output_to_input_does_not_connect_bare_id_cross_tool_even_on_exact_match(
        self,
    ):
        """Denylist: bare ``id`` → ``id`` must not link unrelated tools."""
        reg = _registry(
            _ep(
                "anthill",
                "get_trades",
                required=[("q", "string")],
                response=[("id", "string"), ("name", "string")],
            ),
            _ep(
                "genius",
                "suggestions",
                required=[("id", "string")],
            ),
        )
        tg = build_tool_graph(reg)
        neighbors = tg.neighbors(
            "anthill.get_trades", edge_types=["OUTPUT_TO_INPUT"]
        )
        assert ("genius.suggestions", "OUTPUT_TO_INPUT") not in neighbors

    def test_specific_id_name_still_connects_cross_tool(self):
        reg = _registry(
            _ep(
                "search_api",
                "find",
                required=[("q", "string")],
                response=[("hotel_id", "string")],
            ),
            _ep(
                "booking_api",
                "book",
                required=[("hotel_id", "string")],
            ),
        )
        tg = build_tool_graph(reg)
        neighbors = tg.neighbors("search_api.find", edge_types=["OUTPUT_TO_INPUT"])
        assert ("booking_api.book", "OUTPUT_TO_INPUT") in neighbors

    def test_output_to_input_is_directed(self):
        reg = _registry(
            _ep(
                "hotels",
                "search",
                required=[("city", "string")],
                response=[("id", "string")],
            ),
            _ep(
                "hotels",
                "book",
                required=[("hotel_id", "string")],
            ),
        )
        tg = build_tool_graph(reg)
        # Forward edge exists
        assert ("hotels.book", "OUTPUT_TO_INPUT") in tg.neighbors(
            "hotels.search", edge_types=["OUTPUT_TO_INPUT"]
        )
        # Reverse edge does not
        assert ("hotels.search", "OUTPUT_TO_INPUT") not in tg.neighbors(
            "hotels.book", edge_types=["OUTPUT_TO_INPUT"]
        )

    def test_response_field_enrichment_enables_output_to_input(self):
        # hotels.search has no declared response fields, but its name
        # is "search" so enrichment should inject `id`, which should then
        # enable an OUTPUT_TO_INPUT edge to hotels.book via the bare-id
        # same-tool rule.
        reg = _registry(
            _ep("hotels", "search", required=[("city", "string")]),
            _ep("hotels", "book", required=[("hotel_id", "string")]),
        )
        tg = build_tool_graph(reg)
        assert ("hotels.book", "OUTPUT_TO_INPUT") in tg.neighbors(
            "hotels.search", edge_types=["OUTPUT_TO_INPUT"]
        )

    def test_edge_type_counts(self):
        reg = _registry(
            _ep(
                "hotels",
                "search",
                required=[("city", "string")],
                response=[("id", "string")],
            ),
            _ep("hotels", "book", required=[("hotel_id", "string")]),
        )
        tg = build_tool_graph(reg)
        counts = tg.edge_type_counts()
        assert set(counts.keys()) == set(ALL_EDGE_TYPES)
        # Two endpoints, same tool → 2 SAME_TOOL edges (both directions)
        assert counts["SAME_TOOL"] == 2
        # Forward OUTPUT_TO_INPUT only
        assert counts["OUTPUT_TO_INPUT"] == 1

    def test_neighbors_with_no_edge_type_filter_returns_all(self):
        reg = _registry(
            _ep(
                "hotels",
                "search",
                required=[("city", "string")],
                response=[("id", "string")],
            ),
            _ep("hotels", "book", required=[("hotel_id", "string")]),
        )
        tg = build_tool_graph(reg)
        all_neighbors = tg.neighbors("hotels.search")
        edge_types = {et for _, et in all_neighbors}
        assert "SAME_TOOL" in edge_types
        assert "OUTPUT_TO_INPUT" in edge_types


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: Path):
        reg = _registry(
            _ep("hotels", "search", required=[("city", "string")]),
            _ep("hotels", "book", required=[("hotel_id", "string")]),
        )
        tg = build_tool_graph(reg)
        path = tmp_path / "graph.pkl"
        tg.save(path)

        loaded = ToolGraph.load(path)
        assert len(loaded) == len(tg)
        assert loaded.num_edges == tg.num_edges
        assert loaded.get_endpoint("hotels.search") is not None
        assert loaded.edge_type_counts() == tg.edge_type_counts()


class TestDefaultWeights:
    def test_all_edge_types_have_weights(self):
        assert set(DEFAULT_EDGE_WEIGHTS.keys()) == set(ALL_EDGE_TYPES)

    def test_output_to_input_is_strongest(self):
        assert DEFAULT_EDGE_WEIGHTS["OUTPUT_TO_INPUT"] == max(
            DEFAULT_EDGE_WEIGHTS.values()
        )

    def test_same_category_is_weakest(self):
        assert DEFAULT_EDGE_WEIGHTS["SAME_CATEGORY"] == min(
            DEFAULT_EDGE_WEIGHTS.values()
        )
