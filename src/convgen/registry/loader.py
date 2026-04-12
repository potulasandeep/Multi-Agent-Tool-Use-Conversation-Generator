"""
Loader that normalizes ToolBench JSON dumps into Registry/Tool/Endpoint models.

ToolBench ships thousands of raw JSON tool specs with inconsistent structure:
- Parameter type names vary: "STRING", "str", "String", sometimes missing
- Response schemas are frequently absent
- Descriptions may be missing, empty, or duplicated with the name field
- Some files are malformed or partially corrupt
- Endpoint names can contain slashes, spaces, and other hostile characters

This loader is the anti-corruption layer: it ingests the mess and yields
validated, uniform model instances. Failures at any level (parameter,
endpoint, tool, file) are logged and skipped without cascading. The only
exception raised is FileNotFoundError for a missing data directory, which
is a programmer bug rather than a data-quality issue.

Assumed directory layout:
    <data_dir>/<Category>/<tool>.json
    <data_dir>/<Category>/<subdir>/<tool>.json
The first directory component under data_dir is treated as the category.
Files directly under data_dir get category "Unknown".
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from convgen.registry.models import (
    Endpoint,
    Parameter,
    ParameterType,
    Registry,
    ResponseField,
    Tool,
)

logger = logging.getLogger(__name__)

# Canonicalization map for parameter type strings.
# Anything not in this map becomes "unknown".
_TYPE_MAP: dict[str, ParameterType] = {
    "string": "string", "str": "string", "text": "string", "varchar": "string",
    "integer": "integer", "int": "integer", "long": "integer",
    "int32": "integer", "int64": "integer",
    "number": "number", "float": "number", "double": "number",
    "decimal": "number", "numeric": "number",
    "boolean": "boolean", "bool": "boolean",
    "array": "array", "list": "array",
    "object": "object", "dict": "object", "map": "object", "json": "object",
}

_SLUG_RE = re.compile(r"[^A-Za-z0-9_]+")


def _normalize_type(raw: Any) -> ParameterType:
    """Map a raw type string (or None) to our canonical Literal set.

    None → 'string' (conservative default for missing types).
    Empty/unknown strings → 'unknown' (caller can decide how to handle).
    """
    if raw is None:
        return "string"
    return _TYPE_MAP.get(str(raw).strip().lower(), "unknown")


def _slugify(name: str) -> str:
    """Turn an arbitrary name into a safe [A-Za-z0-9_] identifier fragment."""
    slug = _SLUG_RE.sub("_", name.strip()).strip("_")
    return slug or "unnamed"


def _parse_parameter(raw: Any) -> Parameter | None:
    """Parse one parameter dict; return None if it's unusable."""
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        return Parameter(
            name=name,
            type=_normalize_type(raw.get("type")),
            required=bool(raw.get("required", False)),
            description=str(raw.get("description") or ""),
        )
    except ValidationError as e:
        logger.debug("Skipping parameter %r: %s", name, e)
        return None


def _parse_response_field(raw: Any) -> ResponseField | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        return ResponseField(
            name=name,
            type=_normalize_type(raw.get("type")),
            description=str(raw.get("description") or ""),
        )
    except ValidationError as e:
        logger.debug("Skipping response field %r: %s", name, e)
        return None


def _parse_endpoint(
    raw: dict, tool_name: str, category: str
) -> Endpoint | None:
    """Parse one api_list entry into an Endpoint."""
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        logger.debug("Endpoint in %r has no usable name, skipping", tool_name)
        return None

    endpoint_slug = _slugify(name)
    tool_slug = _slugify(tool_name)
    endpoint_id = f"{tool_slug}.{endpoint_slug}"

    # Required + optional parameters live under different keys in ToolBench.
    # We trust the parent key over any `required` flag inside the entry.
    required_raw = raw.get("required_parameters") or []
    optional_raw = raw.get("optional_parameters") or []

    parameters: list[Parameter] = []
    if isinstance(required_raw, list):
        for p_raw in required_raw:
            parsed = _parse_parameter(p_raw)
            if parsed is not None:
                parameters.append(parsed.model_copy(update={"required": True}))
    if isinstance(optional_raw, list):
        for p_raw in optional_raw:
            parsed = _parse_parameter(p_raw)
            if parsed is not None:
                parameters.append(parsed.model_copy(update={"required": False}))

    # Response schema — ToolBench rarely provides this; try a few common keys.
    response_raw = (
        raw.get("response_schema")
        or raw.get("response")
        or raw.get("output_schema")
        or []
    )
    response_fields: list[ResponseField] = []
    if isinstance(response_raw, list):
        for f_raw in response_raw:
            parsed = _parse_response_field(f_raw)
            if parsed is not None:
                response_fields.append(parsed)

    try:
        return Endpoint(
            id=endpoint_id,
            tool_name=tool_slug,
            name=endpoint_slug,
            description=str(raw.get("description") or ""),
            category=category,
            parameters=parameters,
            response_fields=response_fields,
        )
    except ValidationError as e:
        logger.warning("Failed to build Endpoint %s: %s", endpoint_id, e)
        return None


def _parse_tool(data: dict, category: str) -> Tool | None:
    tool_name = data.get("tool_name") or data.get("name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        logger.warning("Tool file missing tool_name, skipping")
        return None

    api_list = data.get("api_list") or data.get("endpoints") or []
    if not isinstance(api_list, list) or not api_list:
        logger.debug("Tool %r has no api_list, skipping", tool_name)
        return None

    endpoints: list[Endpoint] = []
    for ep_raw in api_list:
        if not isinstance(ep_raw, dict):
            continue
        ep = _parse_endpoint(ep_raw, tool_name=tool_name, category=category)
        if ep is not None:
            endpoints.append(ep)

    if not endpoints:
        logger.debug(
            "Tool %r produced zero valid endpoints, skipping", tool_name
        )
        return None

    try:
        return Tool(
            name=_slugify(tool_name),
            category=category,
            description=str(
                data.get("tool_description") or data.get("description") or ""
            ),
            endpoints=endpoints,
        )
    except ValidationError as e:
        logger.warning("Failed to build Tool %r: %s", tool_name, e)
        return None


def _category_for(path: Path, data_dir: Path) -> str:
    """Category is the first path component under data_dir, or 'Unknown'."""
    try:
        rel = path.relative_to(data_dir)
    except ValueError:
        return "Unknown"
    parts = rel.parts
    if len(parts) <= 1:
        return "Unknown"
    return parts[0]


def load_registry(data_dir: str | Path) -> Registry:
    """
    Walk a directory of ToolBench JSON files and produce a validated Registry.

    Files that cannot be parsed as JSON, tools with no usable endpoints, and
    individual malformed parameters are all logged and skipped. The function
    never raises for data-quality issues — it returns whatever it could
    successfully load.

    Raises:
        FileNotFoundError: if `data_dir` does not exist.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    tools: list[Tool] = []
    skipped = 0
    total = 0

    for json_path in sorted(root.rglob("*.json")):
        total += 1
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", json_path, e)
            skipped += 1
            continue

        if not isinstance(data, dict):
            logger.debug(
                "Skipping %s: top-level JSON is not an object", json_path
            )
            skipped += 1
            continue

        category = _category_for(json_path, root)
        tool = _parse_tool(data, category=category)
        if tool is None:
            skipped += 1
            continue
        tools.append(tool)

    logger.info(
        "Loaded %d tools from %d JSON files (%d skipped) in %s",
        len(tools), total, skipped, root,
    )
    return Registry(tools=tools)
