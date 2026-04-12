"""Tests for the ToolBench loader.

Fixtures are written inline into tmp_path so each test is self-contained
and the specific ToolBench quirks being exercised are visible in the test
body rather than hidden in a separate fixtures/ directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from convgen.registry.loader import (
    _normalize_type,
    _slugify,
    load_registry,
)


# ---------------------- helpers ----------------------

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _well_formed_tool(tool_name: str = "hotels_api") -> dict:
    return {
        "tool_name": tool_name,
        "tool_description": "Search and book hotels worldwide.",
        "api_list": [
            {
                "name": "search",
                "description": "Search for hotels in a city.",
                "required_parameters": [
                    {
                        "name": "city",
                        "type": "string",
                        "description": "City name",
                    },
                ],
                "optional_parameters": [
                    {
                        "name": "max_price",
                        "type": "integer",
                        "description": "Max price/night",
                    },
                ],
            },
            {
                "name": "book",
                "description": "Book a specific hotel.",
                "required_parameters": [
                    {"name": "hotel_id", "type": "string"},
                    {"name": "check_in", "type": "string"},
                ],
                "optional_parameters": [],
            },
        ],
    }


# ---------------------- unit tests on helpers ----------------------

class TestNormalizeType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("STRING", "string"),
            ("str", "string"),
            ("Integer", "integer"),
            ("INT", "integer"),
            ("NUMBER", "number"),
            ("float", "number"),
            ("bool", "boolean"),
            ("ARRAY", "array"),
            ("list", "array"),
            ("dict", "object"),
            (None, "string"),
            ("wibble", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_cases(self, raw, expected):
        assert _normalize_type(raw) == expected


class TestSlugify:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("search", "search"),
            ("locations/v3/search", "locations_v3_search"),
            ("get user-info", "get_user_info"),
            ("  trim me  ", "trim_me"),
            ("!!!", "unnamed"),
        ],
    )
    def test_cases(self, raw, expected):
        assert _slugify(raw) == expected


# ---------------------- integration tests on load_registry ----------------------

class TestLoadRegistry:
    def test_loads_well_formed_tool(self, tmp_path: Path):
        _write_json(tmp_path / "Travel" / "hotels.json", _well_formed_tool())
        reg = load_registry(tmp_path)
        assert len(reg.tools) == 1

        tool = reg.tools[0]
        assert tool.name == "hotels_api"
        assert tool.category == "Travel"
        assert len(tool.endpoints) == 2

        search = reg.get_endpoint("hotels_api.search")
        assert search is not None
        assert search.category == "Travel"
        required_names = [p.name for p in search.required_parameters]
        assert required_names == ["city"]
        assert len(search.parameters) == 2  # city + max_price

    def test_category_from_first_path_component(self, tmp_path: Path):
        _write_json(
            tmp_path / "Travel" / "deep" / "nest" / "hotels.json",
            _well_formed_tool(),
        )
        reg = load_registry(tmp_path)
        assert reg.tools[0].category == "Travel"

    def test_flat_files_get_unknown_category(self, tmp_path: Path):
        _write_json(tmp_path / "hotels.json", _well_formed_tool())
        reg = load_registry(tmp_path)
        assert reg.tools[0].category == "Unknown"

    def test_weird_parameter_types_normalized(self, tmp_path: Path):
        tool = {
            "tool_name": "flights",
            "api_list": [
                {
                    "name": "search",
                    "required_parameters": [
                        {"name": "from_city", "type": "STRING"},
                        {"name": "num_passengers", "type": "INT"},
                        {"name": "one_way", "type": "BOOLEAN"},
                    ],
                    "optional_parameters": [
                        {"name": "extras", "type": "wibble"},  # unknown
                    ],
                }
            ],
        }
        _write_json(tmp_path / "Travel" / "flights.json", tool)
        reg = load_registry(tmp_path)

        ep = reg.get_endpoint("flights.search")
        assert ep is not None
        types = {p.name: p.type for p in ep.parameters}
        assert types == {
            "from_city": "string",
            "num_passengers": "integer",
            "one_way": "boolean",
            "extras": "unknown",
        }

    def test_missing_descriptions_become_empty_strings(self, tmp_path: Path):
        tool = {
            "tool_name": "stocks",
            "api_list": [
                {
                    "name": "quote",
                    "required_parameters": [
                        {"name": "symbol", "type": "string"},
                    ],
                }
            ],
        }
        _write_json(tmp_path / "Finance" / "stocks.json", tool)
        reg = load_registry(tmp_path)

        ep = reg.get_endpoint("stocks.quote")
        assert ep is not None
        assert ep.description == ""
        assert ep.parameters[0].description == ""

    def test_malformed_json_is_skipped_not_fatal(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        (tmp_path / "Weather").mkdir(parents=True)
        (tmp_path / "Weather" / "broken.json").write_text(
            "{not json", encoding="utf-8"
        )
        _write_json(tmp_path / "Travel" / "hotels.json", _well_formed_tool())

        with caplog.at_level(logging.WARNING):
            reg = load_registry(tmp_path)

        # The good file still loaded; the broken one was skipped with a warning.
        assert len(reg.tools) == 1
        assert reg.tools[0].name == "hotels_api"
        assert any("broken.json" in r.message for r in caplog.records)

    def test_tool_with_empty_api_list_is_skipped(self, tmp_path: Path):
        _write_json(
            tmp_path / "Weather" / "empty.json",
            {"tool_name": "empty_tool", "api_list": []},
        )
        reg = load_registry(tmp_path)
        assert len(reg.tools) == 0

    def test_tool_with_no_tool_name_is_skipped(self, tmp_path: Path):
        _write_json(
            tmp_path / "Weather" / "nameless.json",
            {"api_list": [{"name": "x"}]},
        )
        reg = load_registry(tmp_path)
        assert len(reg.tools) == 0

    def test_endpoint_with_slash_in_name_is_slugified(self, tmp_path: Path):
        tool = {
            "tool_name": "locations_api",
            "api_list": [
                {
                    "name": "locations/v3/search",
                    "required_parameters": [{"name": "q", "type": "string"}],
                }
            ],
        }
        _write_json(tmp_path / "Travel" / "loc.json", tool)
        reg = load_registry(tmp_path)

        ep_ids = [ep.id for ep in reg.endpoints]
        assert ep_ids == ["locations_api.locations_v3_search"]

    def test_required_flag_forced_by_parent_key(self, tmp_path: Path):
        # Even if a source entry says required=False inside required_parameters,
        # the loader must force required=True (and vice versa for optional).
        tool = {
            "tool_name": "x",
            "api_list": [
                {
                    "name": "y",
                    "required_parameters": [
                        {"name": "a", "type": "string", "required": False},
                    ],
                    "optional_parameters": [
                        {"name": "b", "type": "string", "required": True},
                    ],
                }
            ],
        }
        _write_json(tmp_path / "Misc" / "x.json", tool)
        reg = load_registry(tmp_path)

        ep = reg.get_endpoint("x.y")
        assert ep is not None
        flags = {p.name: p.required for p in ep.parameters}
        assert flags == {"a": True, "b": False}

    def test_missing_data_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_registry(tmp_path / "does_not_exist")

    def test_empty_data_dir_returns_empty_registry(self, tmp_path: Path):
        reg = load_registry(tmp_path)
        assert len(reg) == 0
        assert reg.tools == []

    def test_multiple_tools_across_categories(self, tmp_path: Path):
        _write_json(tmp_path / "Travel" / "hotels.json", _well_formed_tool("hotels"))
        _write_json(tmp_path / "Travel" / "flights.json", _well_formed_tool("flights"))
        _write_json(tmp_path / "Finance" / "stocks.json", _well_formed_tool("stocks"))
        reg = load_registry(tmp_path)

        assert len(reg.tools) == 3
        assert reg.categories == {"Travel", "Finance"}
        # Flattened endpoint view should include all endpoints from all tools.
        assert len(reg) == 6  # 3 tools × 2 endpoints each
