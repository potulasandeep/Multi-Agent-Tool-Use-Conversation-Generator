"""Tests for the mock executor and session store."""

from __future__ import annotations

import pytest

from convgen.executor.mock import (
    ExecutorError,
    MockExecutor,
    SessionStore,
    _first_token,
    _is_action,
    _is_list_response,
)
from convgen.registry.models import (
    Endpoint,
    Parameter,
    ResponseField,
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


# ---------------------- shape classification ----------------------


class TestShapeClassification:
    @pytest.mark.parametrize(
        "name,is_list",
        [
            ("search", True),
            ("search_hotels", True),
            ("list_users", True),
            ("find_flights", True),
            ("browse_categories", True),
            ("query_data", True),
            ("get_details", False),
            ("book", False),
            ("cancel_booking", False),
        ],
    )
    def test_is_list_response(self, name: str, is_list: bool):
        endpoint = _ep("t", name)
        assert _is_list_response(endpoint) is is_list

    @pytest.mark.parametrize(
        "name,is_action",
        [
            ("book", True),
            ("book_hotel", True),
            ("create_user", True),
            ("cancel_booking", True),
            ("update_profile", True),
            ("delete_account", True),
            ("search", False),
            ("get_details", False),
            ("list_items", False),
        ],
    )
    def test_is_action(self, name: str, is_action: bool):
        endpoint = _ep("t", name)
        assert _is_action(endpoint) is is_action

    def test_first_token_handles_empty(self):
        assert _first_token("") == ""


# ---------------------- SessionStore ----------------------


class TestSessionStore:
    def test_empty_store(self):
        store = SessionStore()
        assert len(store) == 0
        assert store.keys() == []
        assert store.get("anything") == []
        assert store.available_values() == {}
        assert store.call_log == []

    def test_record_and_get(self):
        store = SessionStore()
        store.record("hotel_id", "hot_1234")
        assert store.get("hotel_id") == ["hot_1234"]
        assert "hotel_id" in store

    def test_record_with_tool_namespacing(self):
        store = SessionStore()
        store.record("id", "hot_1234", tool_name="hotels")
        # Both the raw and namespaced keys should be present.
        assert store.get("id") == ["hot_1234"]
        assert store.get("hotels.id") == ["hot_1234"]

    def test_multiple_values_under_same_key(self):
        store = SessionStore()
        store.record("id", "a", tool_name="hotels")
        store.record("id", "b", tool_name="hotels")
        store.record("id", "c", tool_name="hotels")
        assert store.get("id") == ["a", "b", "c"]
        assert store.get("hotels.id") == ["a", "b", "c"]

    def test_clear_resets_state(self):
        store = SessionStore()
        store.record("a", 1)
        store.log_call("ep", {}, {"x": 1})
        store.clear()
        assert len(store) == 0
        assert store.call_log == []

    def test_call_log_records_in_order(self):
        store = SessionStore()
        store.log_call("a.b", {"x": 1}, {"y": 2})
        store.log_call("c.d", {"x": 3}, {"y": 4})
        log = store.call_log
        assert len(log) == 2
        assert log[0]["endpoint"] == "a.b"
        assert log[1]["endpoint"] == "c.d"

    def test_available_values_returns_copy(self):
        store = SessionStore()
        store.record("a", 1)
        snapshot = store.available_values()
        snapshot["a"].append(999)
        # Mutating the snapshot must not affect the store.
        assert store.get("a") == [1]


# ---------------------- ID generation ----------------------


class TestIdGeneration:
    def test_id_format(self):
        executor = MockExecutor(seed=0)
        out = executor._generate_id("hotels")
        assert out.startswith("hot_")
        assert len(out.split("_")[1]) == 4

    def test_id_uses_first_three_chars_of_tool(self):
        executor = MockExecutor(seed=0)
        assert executor._generate_id("flights").startswith("fli_")
        assert executor._generate_id("stocks_market_data").startswith("sto_")

    def test_id_falls_back_for_empty_tool_name(self):
        executor = MockExecutor(seed=0)
        assert executor._generate_id("").startswith("obj_")

    def test_id_deterministic_with_same_seed(self):
        a = MockExecutor(seed=42)._generate_id("hotels")
        b = MockExecutor(seed=42)._generate_id("hotels")
        assert a == b


# ---------------------- execute: validation ----------------------


class TestExecuteValidation:
    def test_missing_required_param_raises(self):
        endpoint = _ep("hotels", "book", required=[("hotel_id", "string")])
        executor = MockExecutor(seed=0)
        with pytest.raises(ExecutorError):
            executor.execute(endpoint, {}, SessionStore())

    def test_extra_args_are_allowed(self):
        endpoint = _ep("hotels", "book", required=[("hotel_id", "string")])
        executor = MockExecutor(seed=0)
        # Should not raise even with unexpected keys.
        executor.execute(endpoint, {"hotel_id": "x", "rubbish": 1}, SessionStore())


# ---------------------- execute: response shapes ----------------------


class TestListResponses:
    def test_search_returns_results_list(self):
        endpoint = _ep(
            "hotels",
            "search",
            required=[("city", "string")],
            response=[("id", "string"), ("name", "string")],
        )
        executor = MockExecutor(seed=0)
        response = executor.execute(endpoint, {"city": "Paris"}, SessionStore())
        assert "results" in response
        assert isinstance(response["results"], list)
        assert 1 <= len(response["results"]) <= 3
        for item in response["results"]:
            assert "id" in item
            assert "name" in item

    def test_list_endpoint_with_empty_fields_falls_back_to_id_name(self):
        endpoint = _ep("hotels", "search", required=[("city", "string")])
        executor = MockExecutor(seed=0)
        response = executor.execute(endpoint, {"city": "Paris"}, SessionStore())
        assert "results" in response
        assert all(
            "id" in item and "name" in item for item in response["results"]
        )


class TestActionResponses:
    def test_book_returns_id_and_status(self):
        endpoint = _ep("hotels", "book", required=[("hotel_id", "string")])
        executor = MockExecutor(seed=0)
        response = executor.execute(endpoint, {"hotel_id": "hot_1"}, SessionStore())
        assert response["status"] == "confirmed"
        assert "book_id" in response
        assert response["book_id"].startswith("hot_")

    def test_cancel_returns_cancel_id(self):
        endpoint = _ep("orders", "cancel_booking", required=[("booking_id", "string")])
        executor = MockExecutor(seed=0)
        response = executor.execute(endpoint, {"booking_id": "ord_1"}, SessionStore())
        assert response["status"] == "confirmed"
        assert "cancel_id" in response

    def test_action_with_declared_fields_uses_them(self):
        endpoint = _ep(
            "hotels",
            "book",
            required=[("hotel_id", "string")],
            response=[("booking_id", "string"), ("total_price", "number")],
        )
        executor = MockExecutor(seed=0)
        response = executor.execute(endpoint, {"hotel_id": "hot_1"}, SessionStore())
        assert "booking_id" in response
        assert "total_price" in response
        assert isinstance(response["total_price"], float)


class TestSingleResponses:
    def test_get_details_returns_flat_fields(self):
        endpoint = _ep(
            "hotels",
            "get_details",
            required=[("hotel_id", "string")],
            response=[("rating", "number"), ("city", "string")],
        )
        executor = MockExecutor(seed=0)
        response = executor.execute(endpoint, {"hotel_id": "hot_1"}, SessionStore())
        assert "results" not in response
        assert "rating" in response
        assert "city" in response

    def test_empty_fields_returns_synthetic_id_and_name(self):
        """Previously returned ``{'status': 'ok'}`` — ungroundable for later turns."""
        endpoint = _ep("misc", "ping", required=[])
        executor = MockExecutor(seed=0)
        response = executor.execute(endpoint, {}, SessionStore())
        assert "id" in response
        assert "name" in response
        assert str(response["id"]).startswith("mis_")


# ---------------------- value generation semantics ----------------------


class TestSemanticValueGeneration:
    def test_email_field_gets_email(self):
        endpoint = _ep(
            "users",
            "get_profile",
            required=[("user_id", "string")],
            response=[("contact_email", "string")],
        )
        executor = MockExecutor(seed=0)
        response = executor.execute(endpoint, {"user_id": "u1"}, SessionStore())
        assert "@" in response["contact_email"]

    def test_city_field_gets_city(self):
        endpoint = _ep(
            "places",
            "get_info",
            required=[("place_id", "string")],
            response=[("city", "string")],
        )
        executor = MockExecutor(seed=0)
        response = executor.execute(endpoint, {"place_id": "p1"}, SessionStore())
        assert isinstance(response["city"], str) and len(response["city"]) > 0

    def test_price_field_gets_number(self):
        endpoint = _ep(
            "items",
            "get_info",
            required=[("item_id", "string")],
            response=[("price", "string")],  # type lies, name wins
        )
        executor = MockExecutor(seed=0)
        response = executor.execute(endpoint, {"item_id": "i1"}, SessionStore())
        assert isinstance(response["price"], float)
        assert 10 <= response["price"] <= 1000


# ---------------------- the key test: chained grounding ----------------------


class TestChainedGrounding:
    """The flow that justifies the entire SessionStore design.

    Step 1: hotels.search runs and produces fake hotel records.
    Step 2: We pull a hotel id from the session and pass it to hotels.book.
    Step 3: hotels.book succeeds, and its booking id also appears in the session.

    This is the data flow the orchestrator will reproduce in Step 7,
    except the orchestrator will hand the available values to an LLM
    instead of picking the first one programmatically.
    """

    def test_search_then_book_chains_through_session(self):
        search_endpoint = _ep(
            "hotels",
            "search",
            required=[("city", "string")],
            response=[("id", "string"), ("name", "string"), ("price", "number")],
        )
        book_endpoint = _ep(
            "hotels",
            "book",
            required=[("hotel_id", "string"), ("check_in", "string")],
        )
        executor = MockExecutor(seed=0)
        session = SessionStore()

        # Step 1: search produces 1-3 hotels with fake ids.
        first_response = executor.execute(search_endpoint, {"city": "Paris"}, session)
        assert "results" in first_response
        produced_ids = session.get("id")
        assert len(produced_ids) >= 1

        # The orchestrator would now pick one of these ids and pass it
        # as `hotel_id` to the next call. We do that programmatically.
        chosen = produced_ids[0]
        second_response = executor.execute(
            book_endpoint,
            {"hotel_id": chosen, "check_in": "2026-04-11"},
            session,
        )
        assert second_response["status"] == "confirmed"
        assert "book_id" in second_response

        # Both the search id and the booking id should now be in the
        # session, the latter under both raw and namespaced keys.
        assert "id" in session
        assert "book_id" in session
        assert "hotels.book_id" in session

        # The call log should record both calls in order.
        log = session.call_log
        assert [call["endpoint"] for call in log] == ["hotels.search", "hotels.book"]


# ---------------------- determinism ----------------------


class TestDeterminism:
    def test_same_seed_same_responses(self):
        endpoint = _ep(
            "hotels",
            "search",
            required=[("city", "string")],
            response=[("id", "string"), ("name", "string")],
        )

        def run() -> dict:
            executor = MockExecutor(seed=99)
            session = SessionStore()
            return executor.execute(endpoint, {"city": "Paris"}, session)

        first = run()
        second = run()
        assert first == second

    def test_different_seeds_typically_differ(self):
        endpoint = _ep(
            "hotels",
            "search",
            required=[("city", "string")],
            response=[("id", "string")],
        )
        results = set()
        for seed in range(10):
            response = MockExecutor(seed=seed).execute(
                endpoint, {"city": "Paris"}, SessionStore()
            )
            results.add(tuple(item["id"] for item in response["results"]))
        assert len(results) > 1
