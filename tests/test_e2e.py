"""
End-to-end test: build -> generate >=100 conversations -> judge -> assert quality.

This test is gated behind `@pytest.mark.e2e` because it hits a real LLM
provider. Run with:

    pytest -m e2e -v

Quality gate: mean judge score uses ``QUALITY_THRESHOLD`` (3.4), set from the
diversity experiment at n=100 on a 240-tool ToolBench subset (both runs ~3.44
mean), minus a small buffer for run-to-run variance. Grounding fidelity is
asserted separately at ``GROUNDING_FLOOR`` because the mock executor caps
``grounding_fidelity`` near ~2.5–2.6 regardless of mean score (see DESIGN §14.1).

The optional ``data/toolbench`` e2e uses ``REAL_TOOLBENCH_MEAN_FLOOR`` (3.2): full
ToolBench dumps are harsher than the curated 240-tool subset where both diversity
runs averaged ~3.44 (DESIGN §12.2).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from convgen.agents.assistant import AssistantAgent
from convgen.agents.planner import PlannerAgent
from convgen.agents.user import UserAgent
from convgen.executor.mock import MockExecutor
from convgen.graph.builder import build_tool_graph
from convgen.graph.sampler import ConstrainedSampler, cli_generation_slot_constraints
from convgen.io import read_dataset, write_conversation
from convgen.judge.judge import Judge
from convgen.judge.repair import RepairConfig, RepairLoop
from convgen.llm.client import LLMError, make_client
from convgen.orchestrator import Orchestrator
from convgen.registry.loader import load_registry
from convgen.steering.metrics import (
    mean_judge_scores,
    multi_step_ratio,
    multi_tool_ratio,
)
from convgen.steering.tracker import CoverageTracker

QUALITY_THRESHOLD = 3.4
REAL_TOOLBENCH_MEAN_FLOOR = 3.2
GROUNDING_FLOOR = 2.4
MULTI_STEP_REQUIREMENT = 0.50
N_SAMPLES = 100
SEED = 42
MIN_CHAIN = 2
MAX_CHAIN = 5


def _write_tool(path: Path, tool_name: str, endpoints: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tool_name": tool_name,
                "tool_description": f"{tool_name} api",
                "api_list": endpoints,
            }
        ),
        encoding="utf-8",
    )


def _build_e2e_toolbench(root: Path) -> None:
    _write_tool(
        root / "Travel" / "hotels.json",
        "hotels",
        [
            {
                "name": "search",
                "description": "Search hotels in a city",
                "required_parameters": [{"name": "city", "type": "string"}],
                "optional_parameters": [{"name": "max_price", "type": "integer"}],
            },
            {
                "name": "get_details",
                "description": "Get hotel details",
                "required_parameters": [{"name": "hotel_id", "type": "string"}],
                "optional_parameters": [],
            },
            {
                "name": "book",
                "description": "Book a hotel",
                "required_parameters": [
                    {"name": "hotel_id", "type": "string"},
                    {"name": "check_in", "type": "string"},
                ],
                "optional_parameters": [],
            },
        ],
    )
    _write_tool(
        root / "Travel" / "flights.json",
        "flights",
        [
            {
                "name": "search",
                "description": "Search flights",
                "required_parameters": [
                    {"name": "from_city", "type": "string"},
                    {"name": "to_city", "type": "string"},
                ],
                "optional_parameters": [],
            },
            {
                "name": "book",
                "description": "Book a flight",
                "required_parameters": [{"name": "flight_id", "type": "string"}],
                "optional_parameters": [],
            },
        ],
    )
    _write_tool(
        root / "Weather" / "forecast.json",
        "weather",
        [
            {
                "name": "forecast",
                "description": "Get weather forecast",
                "required_parameters": [
                    {"name": "city", "type": "string"},
                    {"name": "date", "type": "string"},
                ],
                "optional_parameters": [],
            },
        ],
    )
    _write_tool(
        root / "Finance" / "stocks.json",
        "stocks",
        [
            {
                "name": "quote",
                "description": "Get stock quote",
                "required_parameters": [{"name": "symbol", "type": "string"}],
                "optional_parameters": [],
            },
            {
                "name": "history",
                "description": "Stock price history",
                "required_parameters": [
                    {"name": "symbol", "type": "string"},
                    {"name": "days", "type": "integer"},
                ],
                "optional_parameters": [],
            },
        ],
    )
    _write_tool(
        root / "Food" / "restaurants.json",
        "restaurants",
        [
            {
                "name": "search",
                "description": "Find restaurants",
                "required_parameters": [{"name": "city", "type": "string"}],
                "optional_parameters": [{"name": "cuisine", "type": "string"}],
            },
            {
                "name": "reserve",
                "description": "Reserve a table",
                "required_parameters": [
                    {"name": "restaurant_id", "type": "string"},
                    {"name": "party_size", "type": "integer"},
                ],
                "optional_parameters": [],
            },
        ],
    )
    _write_tool(
        root / "Media" / "movies.json",
        "movies",
        [
            {
                "name": "search",
                "description": "Search movies",
                "required_parameters": [{"name": "query", "type": "string"}],
                "optional_parameters": [],
            },
            {
                "name": "get_showtimes",
                "description": "Get showtimes for a movie",
                "required_parameters": [
                    {"name": "movie_id", "type": "string"},
                    {"name": "city", "type": "string"},
                ],
                "optional_parameters": [],
            },
        ],
    )


@pytest.mark.e2e
def test_end_to_end_pipeline(tmp_path: Path):
    """Hermetic e2e run using an inline ToolBench-shaped fixture."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        pytest.skip("No LLM provider configured")

    toolbench = tmp_path / "toolbench"
    _build_e2e_toolbench(toolbench)
    _run_pipeline_for_data_dir(toolbench, tmp_path / "dataset.jsonl")


def _run_pipeline_for_data_dir(
    data_dir: Path,
    out_path: Path,
    *,
    mean_threshold: float = QUALITY_THRESHOLD,
) -> None:
    registry = load_registry(data_dir)
    assert len(registry) > 0

    graph = build_tool_graph(registry)
    assert len(graph) > 0

    try:
        llm = make_client()
    except LLMError as e:
        pytest.skip(f"LLM client could not be constructed: {e}")

    rng = random.Random(SEED)
    tracker = CoverageTracker()
    sampler = ConstrainedSampler(graph, rng=rng, bias_provider=tracker)

    orchestrator = Orchestrator(
        planner=PlannerAgent(llm),
        user=UserAgent(llm),
        assistant=AssistantAgent(llm),
        executor=MockExecutor(seed=SEED),
        clarification_rate=0.3,
    )
    judge = Judge(llm)
    repair = RepairLoop(
        orchestrator=orchestrator,
        judge=judge,
        config=RepairConfig(threshold=QUALITY_THRESHOLD, max_repairs=2),
    )

    for i in range(N_SAMPLES):
        chain = sampler.sample(
            cli_generation_slot_constraints(
                i,
                N_SAMPLES,
                min_length=MIN_CHAIN,
                max_length=MAX_CHAIN,
            )
        )
        conv = repair.run(
            chain=chain,
            seed=SEED * 100000 + i,
            conversation_id=f"conv_{i:05d}",
        )
        tracker.record(chain)
        write_conversation(out_path, conv)

    convs = list(read_dataset(out_path))
    assert len(convs) == N_SAMPLES

    means = mean_judge_scores(convs)
    assert means, "No conversations had judge scores attached"
    overall_mean = means.get("mean", 0.0)
    assert overall_mean >= mean_threshold, (
        f"mean judge {overall_mean:.3f} below threshold {mean_threshold}"
    )

    ms = multi_step_ratio(convs, min_steps=3)
    mt = multi_tool_ratio(convs, min_tools=2)
    assert ms >= MULTI_STEP_REQUIREMENT or mt >= MULTI_STEP_REQUIREMENT

    grounding = means.get("grounding_fidelity", 0.0)
    assert grounding >= GROUNDING_FLOOR, (
        f"grounding_fidelity {grounding} below mock-data floor {GROUNDING_FLOOR}"
    )


@pytest.mark.e2e
def test_end_to_end_with_real_toolbench(tmp_path: Path) -> None:
    """Real-data e2e run using `data/toolbench` when present."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        pytest.skip("No LLM provider configured")

    real_data = Path("data/toolbench")
    json_files = list(real_data.rglob("*.json")) if real_data.exists() else []
    if len(json_files) < 10:
        pytest.skip(
            f"data/toolbench has only {len(json_files)} JSON files; "
            "add real ToolBench data to run this test"
        )
    _run_pipeline_for_data_dir(
        real_data,
        tmp_path / "dataset_real_toolbench.jsonl",
        mean_threshold=REAL_TOOLBENCH_MEAN_FLOOR,
    )


@pytest.mark.e2e
def test_end_to_end_pipeline_hermetic_fixture(tmp_path: Path) -> None:
    """Alias test name retained for compatibility with earlier docs."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        pytest.skip("No LLM provider configured")
    toolbench = tmp_path / "toolbench_fixture"
    _build_e2e_toolbench(toolbench)
    _run_pipeline_for_data_dir(toolbench, tmp_path / "dataset_fixture.jsonl")

