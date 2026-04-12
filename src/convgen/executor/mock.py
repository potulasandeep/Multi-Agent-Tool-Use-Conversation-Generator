"""
Offline mock executor for tool-call generation.

The MockExecutor produces schema-consistent fake responses for ToolBench
endpoints without ever calling a real API. The SessionStore tracks every
value the executor has produced within a single conversation so that
later tool calls can reference real prior values.

This module is the backbone of the "grounding via mechanics" design:
the orchestrator (Step 7) injects `session.available_values()` into the
assistant agent's prompt, so the assistant picks IDs from a literal list
rather than hallucinating them. Without this module, grounding would be
a prompting problem and we would need a much stronger model to get the
same fidelity.

Key design choices documented inline; see DESIGN.md for the longer form.

- SessionStore stores values as lists (1-N produced values per field).
- Each value is recorded under both the raw field name and a
  tool-namespaced key (e.g. "id" and "hotels.id") so the orchestrator
  can disambiguate when multiple tools produce same-named fields.
- Required-parameter validation happens here; argument *value* validation
  (i.e. "did this hotel_id come from the session?") is left to the judge.
- Response shape is decided by endpoint name verb, not declared schema:
  list verbs produce {"results": [...]} ; action verbs produce a flat
  {"<verb>_id": ..., "status": "confirmed"}; everything else is a flat
  dict of generated response fields.
- Determinism: Faker and a separate random.Random are both seeded from
  the same constructor seed.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from faker import Faker

from convgen.registry.models import Endpoint, ResponseField

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Endpoint shape classification
# --------------------------------------------------------------------------

# Verbs that imply the endpoint returns multiple entities.
_LIST_RESPONSE_VERBS: frozenset[str] = frozenset(
    {"search", "list", "find", "browse", "query", "all"}
)

# Verbs that imply the endpoint creates or mutates a single entity.
_ACTION_VERBS: frozenset[str] = frozenset(
    {
        "book",
        "create",
        "submit",
        "register",
        "add",
        "post",
        "send",
        "schedule",
        "reserve",
        "cancel",
        "delete",
        "update",
        "make",
        "place",
        "set",
        "remove",
    }
)


def _first_token(name: str) -> str:
    return name.split("_")[0].lower() if name else ""


def _is_list_response(endpoint: Endpoint) -> bool:
    return _first_token(endpoint.name) in _LIST_RESPONSE_VERBS


def _is_action(endpoint: Endpoint) -> bool:
    return _first_token(endpoint.name) in _ACTION_VERBS


def _entity_value_generators(
    faker: Faker, rng: random.Random
) -> dict[str, Any]:
    """Entity keyword -> value generator used for semantic name fields."""
    return {
        "airport": lambda: f"{faker.city()} International Airport",
        "airline": lambda: f"{faker.company()} Airways",
        "airlines": lambda: f"{faker.company()} Airways",
        "country": faker.country,
        "city": faker.city,
        "company": faker.company,
        "business": faker.company,
        "movie": lambda: faker.catch_phrase().title(),
        "film": lambda: faker.catch_phrase().title(),
        "drama": lambda: faker.catch_phrase().title(),
        "show": lambda: faker.catch_phrase().title(),
        "song": lambda: faker.catch_phrase().title(),
        "track": lambda: faker.catch_phrase().title(),
        "album": lambda: faker.catch_phrase().title(),
        "lyric": lambda: faker.sentence(nb_words=6),
        "artist": faker.name,
        "actor": faker.name,
        "trader": faker.name,
        "user": faker.name,
        "author": faker.name,
        "book": lambda: faker.catch_phrase().title(),
        "article": lambda: faker.catch_phrase().title(),
        "news": lambda: faker.sentence(nb_words=8),
        "genre": lambda: faker.word().capitalize(),
        "tag": lambda: faker.word().capitalize(),
        "category": lambda: faker.word().capitalize(),
        "stock": lambda: "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=4)),
        "symbol": lambda: "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=4)),
        "trade": lambda: faker.bs().title(),
        "market": faker.city,
        "currency": lambda: rng.choice(["USD", "EUR", "GBP", "JPY", "CHF"]),
    }


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class ExecutorError(Exception):
    """Raised when an endpoint cannot be executed (e.g., missing required
    parameters). Indicates a generation-time bug, not a data quality
    issue - the orchestrator may catch this and trigger repair."""


# --------------------------------------------------------------------------
# SessionStore
# --------------------------------------------------------------------------


class SessionStore:
    """Per-conversation memory of every value the executor has produced.

    Values are stored as lists keyed by both their raw field name and
    a tool-namespaced version. The orchestrator reads from here when
    building the assistant agent's prompt context for the next turn.
    """

    def __init__(self) -> None:
        self._values: dict[str, list[Any]] = {}
        self._call_log: list[dict[str, Any]] = []

    # -- value recording / retrieval -----------------------------------

    def record(
        self,
        field_name: str,
        value: Any,
        tool_name: str | None = None,
    ) -> None:
        """Store a value under its raw field name and (optionally) under
        a tool-namespaced key like 'hotels.id'."""
        self._values.setdefault(field_name, []).append(value)
        if tool_name:
            ns_key = f"{tool_name}.{field_name}"
            self._values.setdefault(ns_key, []).append(value)

    def get(self, field_name: str) -> list[Any]:
        """Return all values recorded under a given key (raw or namespaced)."""
        return list(self._values.get(field_name, []))

    def available_values(self) -> dict[str, list[Any]]:
        """Snapshot of every recorded value. Used by the orchestrator to
        build the assistant prompt's 'available values' block."""
        return {k: list(v) for k, v in self._values.items()}

    def keys(self) -> list[str]:
        return list(self._values.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __len__(self) -> int:
        return len(self._values)

    # -- call logging --------------------------------------------------

    def log_call(
        self,
        endpoint_id: str,
        arguments: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        self._call_log.append(
            {
                "endpoint": endpoint_id,
                "arguments": arguments,
                "response": response,
            }
        )

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return list(self._call_log)

    # -- lifecycle -----------------------------------------------------

    def clear(self) -> None:
        self._values.clear()
        self._call_log.clear()


# --------------------------------------------------------------------------
# MockExecutor
# --------------------------------------------------------------------------


class MockExecutor:
    """Produces schema-consistent fake responses for endpoint calls and
    populates the SessionStore as a side effect.

    Determinism: a single `seed` parameter seeds both Faker and an
    internal random.Random, so two executors constructed with the same
    seed produce identical responses given identical call sequences.
    """

    def __init__(self, seed: int = 0) -> None:
        self.faker = Faker()
        self.faker.seed_instance(seed)
        self._rng = random.Random(seed)
        self._entity_generators = _entity_value_generators(
            self.faker, self._rng
        )

    # -- public API ----------------------------------------------------

    def execute(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        session: SessionStore,
    ) -> dict[str, Any]:
        """Execute one endpoint call and return its mock response.

        Side effects:
            - Records every primitive value in the response into `session`.
            - Appends a call entry to `session.call_log`.

        Raises:
            ExecutorError: if a required parameter is missing.
        """
        missing = [
            parameter.name
            for parameter in endpoint.required_parameters
            if parameter.name not in arguments
        ]
        if missing:
            raise ExecutorError(
                f"Missing required parameters for {endpoint.id}: {missing}"
            )

        if _is_list_response(endpoint):
            response = self._generate_list_response(endpoint)
        elif _is_action(endpoint):
            response = self._generate_action_response(endpoint)
        else:
            response = self._generate_single_response(endpoint)

        self._record_response(endpoint, response, session)
        session.log_call(endpoint.id, arguments, response)
        return response

    # -- response shape generators -------------------------------------

    def _generate_list_response(self, endpoint: Endpoint) -> dict[str, Any]:
        """List endpoints return {'results': [1-3 records]}.

        If the endpoint declares no response fields, fall back to
        synthetic 'id' and 'name' so the result is still useful to
        downstream chains.
        """
        n_results = self._rng.randint(1, 3)
        fields = endpoint.response_fields or [
            ResponseField(name="id", type="string"),
            ResponseField(name="name", type="string"),
        ]
        items: list[dict[str, Any]] = []
        for _ in range(n_results):
            item = {
                field.name: self._generate_value(
                    field.name, field.type, endpoint.tool_name
                )
                for field in fields
            }
            items.append(item)
        return {"results": items}

    def _generate_action_response(self, endpoint: Endpoint) -> dict[str, Any]:
        """Action endpoints return a flat dict with a new entity ID and
        a status field. Examples:

            hotels.book   -> {'book_id': 'hot_8821', 'status': 'confirmed'}
            orders.cancel -> {'cancel_id': 'ord_3391', 'status': 'confirmed'}

        If response_fields is declared, those are used instead.
        """
        if endpoint.response_fields:
            return {
                field.name: self._generate_value(
                    field.name, field.type, endpoint.tool_name
                )
                for field in endpoint.response_fields
            }
        verb = _first_token(endpoint.name) or "action"
        return {
            f"{verb}_id": self._generate_id(endpoint.tool_name),
            "status": "confirmed",
        }

    def _generate_single_response(self, endpoint: Endpoint) -> dict[str, Any]:
        """Default shape: a flat dict mapping each declared response field
        to a generated value.

        If no response fields are declared, fall back to synthetic ``id``
        and ``name`` so the response is still groundable. A bare
        ``{'status': 'ok'}`` made the endpoint invisible to later turns.
        """
        fields = endpoint.response_fields or [
            ResponseField(name="id", type="string"),
            ResponseField(name="name", type="string"),
        ]
        return {
            f.name: self._generate_value(f.name, f.type, endpoint.tool_name)
            for f in fields
        }

    # -- value generation ----------------------------------------------

    def _generate_value(
        self, field_name: str, field_type: str, tool_name: str
    ) -> Any:
        """Pick a fake value for one field. Uses semantic hints from the
        field name first, then falls back to type-based generation."""
        normalized_name = field_name.lower()

        # Semantic hints (checked before type to give nicer values).
        if normalized_name == "id" or normalized_name.endswith("_id"):
            return self._generate_id(tool_name)
        if "email" in normalized_name:
            return self.faker.email()
        if (
            normalized_name == "name"
            or normalized_name.endswith("_name")
            or normalized_name == "title"
            or normalized_name.endswith("_title")
        ):
            prefix = normalized_name
            if normalized_name.endswith("_name"):
                prefix = normalized_name[: -len("_name")]
            elif normalized_name.endswith("_title"):
                prefix = normalized_name[: -len("_title")]
            if prefix and prefix in self._entity_generators:
                return self._entity_generators[prefix]()
            if "company" in normalized_name or "tool" in normalized_name:
                return self.faker.company()
            return self.faker.name()
        if (
            normalized_name == "count"
            or normalized_name.endswith("_count")
            or normalized_name == "total"
        ):
            return self._rng.randint(5, 500)
        if "city" in normalized_name:
            return self.faker.city()
        if "country" in normalized_name:
            return self.faker.country()
        if "address" in normalized_name:
            return self.faker.street_address()
        if "phone" in normalized_name:
            return self.faker.phone_number()
        if "url" in normalized_name or "link" in normalized_name:
            return self.faker.url()
        if (
            "date" in normalized_name
            or normalized_name.endswith("_at")
            or normalized_name in {"check_in", "check_out"}
        ):
            return self.faker.date()
        if (
            "price" in normalized_name
            or "cost" in normalized_name
            or "amount" in normalized_name
            or "fee" in normalized_name
        ):
            return round(self._rng.uniform(10, 1000), 2)
        if "rating" in normalized_name or "score" in normalized_name:
            return round(self._rng.uniform(1, 5), 1)
        if "currency" in normalized_name:
            return self._rng.choice(["USD", "EUR", "GBP", "JPY"])
        if "status" in normalized_name:
            return self._rng.choice(["confirmed", "pending", "available"])

        # Type-based fallback.
        if field_type == "integer":
            return self._rng.randint(1, 10000)
        if field_type == "number":
            return round(self._rng.uniform(0, 1000), 2)
        if field_type == "boolean":
            return self._rng.choice([True, False])
        if field_type == "array":
            return []
        if field_type == "object":
            return {}
        return self.faker.word()

    def _generate_id(self, tool_name: str) -> str:
        """Generate a short tool-prefixed ID like 'hot_8821'."""
        clean = "".join(char for char in tool_name if char.isalnum()).lower()
        prefix = clean[:3] if clean else "obj"
        return f"{prefix}_{self._rng.randint(1000, 9999)}"

    # -- session recording ---------------------------------------------

    def _record_response(
        self,
        endpoint: Endpoint,
        response: dict[str, Any],
        session: SessionStore,
    ) -> None:
        """Walk the response and record every primitive value into the
        session under both its raw field name and a tool-namespaced key.

        Recursive into nested dicts and lists so {"results": [{"id":...}]}
        gets each id stored individually under "id".
        """

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (str, int, float, bool)):
                        session.record(key, value, tool_name=endpoint.tool_name)
                    elif isinstance(value, (dict, list)):
                        walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(response)
