"""Smoke tests for the registry data model."""

import pytest
from pydantic import ValidationError

from convgen.registry.models import (
    Endpoint,
    Parameter,
    Registry,
    ResponseField,
    Tool,
)


def _make_endpoint(
    tool: str = "hotels_api",
    name: str = "search",
    category: str = "Travel",
) -> Endpoint:
    return Endpoint(
        id=f"{tool}.{name}",
        tool_name=tool,
        name=name,
        category=category,
        parameters=[
            Parameter(name="city", type="string", required=True),
            Parameter(name="max_price", type="integer", required=False),
        ],
        response_fields=[
            ResponseField(name="id", type="string"),
            ResponseField(name="name", type="string"),
            ResponseField(name="price", type="number"),
        ],
    )


class TestParameter:
    def test_defaults(self):
        p = Parameter(name="city")
        assert p.type == "string"
        assert p.required is False
        assert p.description == ""

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            Parameter(name="")

    def test_whitespace_name_stripped(self):
        p = Parameter(name="  city  ")
        assert p.name == "city"

    def test_unknown_type_rejected(self):
        # The loader will normalize unknown types to "string" before
        # constructing Parameter; the model itself refuses junk.
        with pytest.raises(ValidationError):
            Parameter(name="city", type="str")  # type: ignore[arg-type]


class TestEndpoint:
    def test_instantiates_with_minimal_fields(self):
        ep = Endpoint(
            id="tool.ep",
            tool_name="tool",
            name="ep",
            category="Travel",
        )
        assert ep.parameters == []
        assert ep.response_fields == []

    def test_id_must_contain_dot(self):
        with pytest.raises(ValidationError):
            Endpoint(
                id="no_dot_here",
                tool_name="x",
                name="y",
                category="Travel",
            )

    def test_required_parameters_property(self):
        ep = _make_endpoint()
        names = [p.name for p in ep.required_parameters]
        assert names == ["city"]


class TestRegistry:
    def test_empty_registry(self):
        reg = Registry()
        assert len(reg) == 0
        assert reg.endpoints == []
        assert reg.categories == set()

    def test_flattened_endpoints(self):
        tool = Tool(
            name="hotels_api",
            category="Travel",
            endpoints=[
                _make_endpoint(name="search"),
                _make_endpoint(name="book"),
            ],
        )
        reg = Registry(tools=[tool])
        assert len(reg) == 2
        assert {ep.name for ep in reg.endpoints} == {"search", "book"}

    def test_get_endpoint_by_id(self):
        tool = Tool(
            name="hotels_api",
            category="Travel",
            endpoints=[_make_endpoint(name="search")],
        )
        reg = Registry(tools=[tool])
        ep = reg.get_endpoint("hotels_api.search")
        assert ep is not None
        assert ep.name == "search"

    def test_get_endpoint_missing_returns_none(self):
        reg = Registry()
        assert reg.get_endpoint("nope.nothing") is None

    def test_categories_aggregated(self):
        reg = Registry(
            tools=[
                Tool(name="a", category="Travel"),
                Tool(name="b", category="Finance"),
                Tool(name="c", category="Travel"),
            ]
        )
        assert reg.categories == {"Travel", "Finance"}
