"""
Internal data model for normalized ToolBench tool specifications.

This module defines the single source of truth for the shape of tools,
endpoints, parameters, and response fields that flow through the rest of
the pipeline. The loader's job is to convert messy ToolBench JSON into
instances of these models; everything downstream (graph, sampler, executor,
agents, judge) consumes these types and relies on them being clean.

Design notes:
- Endpoint is the unit of sampling, not Tool. Chains are sequences of
  endpoints; Tool exists mostly as a grouping/metadata container.
- `category` is denormalized onto Endpoint so the sampler and graph layer
  can filter without joins.
- Parameter types are restricted to a Literal set; the loader normalizes
  unknown types to "string".
- ResponseField is a first-class model because OUTPUT_TO_INPUT edge
  detection in the graph builder depends on it being structured.
- Endpoint.id is a stable human-readable string like "hotels_api.search".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ParameterType = Literal[
    "string", "integer", "number", "boolean", "array", "object", "unknown"
]


class Parameter(BaseModel):
    """A single input parameter on an endpoint."""

    name: str
    type: ParameterType = "string"
    required: bool = False
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Parameter name must be non-empty")
        return v.strip()


class ResponseField(BaseModel):
    """A single field in an endpoint's response schema.

    Used by the graph builder to detect OUTPUT_TO_INPUT edges: an edge
    exists from endpoint A to endpoint B if B has a required parameter
    whose name matches (loosely) one of A's response field names.
    """

    name: str
    type: ParameterType = "string"
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ResponseField name must be non-empty")
        return v.strip()


class Endpoint(BaseModel):
    """A single callable endpoint on a tool.

    This is the fundamental unit the rest of the pipeline operates on:
    the graph's nodes are endpoints, the sampler returns endpoints, the
    executor runs endpoints.
    """

    id: str = Field(
        description="Stable human-readable ID, e.g. 'hotels_api.search'"
    )
    tool_name: str
    name: str = Field(description="Endpoint name within its tool, e.g. 'search'")
    description: str = ""
    category: str = Field(
        description="Denormalized from the parent Tool for sampler convenience"
    )
    parameters: list[Parameter] = Field(default_factory=list)
    response_fields: list[ResponseField] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if "." not in v:
            raise ValueError(
                f"Endpoint.id must be of the form '<tool>.<endpoint>', got {v!r}"
            )
        return v

    @property
    def required_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters if p.required]


class Tool(BaseModel):
    """A ToolBench tool — a collection of related endpoints."""

    name: str
    category: str
    description: str = ""
    endpoints: list[Endpoint] = Field(default_factory=list)


class Registry(BaseModel):
    """An in-memory collection of all loaded tools.

    Thin wrapper: holds tools plus an endpoint-by-id index for O(1)
    lookup. Anything fancier (filtering, querying) belongs in the graph
    layer, not here.
    """

    tools: list[Tool] = Field(default_factory=list)

    @property
    def endpoints(self) -> list[Endpoint]:
        """Flattened view of every endpoint across every tool."""
        return [ep for tool in self.tools for ep in tool.endpoints]

    @property
    def categories(self) -> set[str]:
        return {tool.category for tool in self.tools}

    def get_endpoint(self, endpoint_id: str) -> Endpoint | None:
        """Look up an endpoint by its stable ID."""
        for ep in self.endpoints:
            if ep.id == endpoint_id:
                return ep
        return None

    def __len__(self) -> int:
        return len(self.endpoints)
