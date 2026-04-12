"""Tests for the CLI commands. No real LLM calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from convgen.cli import app
from convgen.llm import FakeLLMClient

runner = CliRunner()


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


@pytest.fixture
def toolbench_dir(tmp_path: Path) -> Path:
    root = tmp_path / "toolbench"
    _write_tool(
        root / "Travel" / "hotels.json",
        "hotels",
        [
            {
                "name": "search",
                "description": "Search hotels",
                "required_parameters": [{"name": "city", "type": "string"}],
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
        ],
    )
    return root


def _canned_fake() -> FakeLLMClient:
    return FakeLLMClient(
        structured_responses={
            "Design a realistic user scenario": {
                "user_intent": "I want to book a hotel in Paris for next weekend",
                "persona": "A weekend traveler",
                "needs_clarification": False,
                "clarification_question": "",
                "withheld_parameters": [],
            },
            "`hotels.search`": {
                "action": "tool_call",
                "clarification": "",
                "tool_endpoint": "",
                "tool_arguments": {"city": "Paris"},
                "final_answer": "",
            },
            "`hotels.book`": {
                "action": "tool_call",
                "clarification": "",
                "tool_endpoint": "",
                "tool_arguments": {
                    "hotel_id": "hot_1234",
                    "check_in": "2026-04-11",
                },
                "final_answer": "",
            },
            "`flights.search`": {
                "action": "tool_call",
                "clarification": "",
                "tool_endpoint": "",
                "tool_arguments": {
                    "from_city": "Paris",
                    "to_city": "London",
                },
                "final_answer": "",
            },
            "action=final_answer and write": {
                "action": "final_answer",
                "clarification": "",
                "tool_endpoint": "",
                "tool_arguments": {},
                "final_answer": "All set! Your booking is confirmed.",
            },
            "Score this conversation": {
                "tool_correctness": 4.5,
                "grounding_fidelity": 4.5,
                "naturalness": 4.0,
                "task_completion": 4.5,
                "reasoning": "Looks good across all dimensions.",
                "failing_turn_index": None,
            },
        },
        text_responses={
            "OPENING message": "Find me a hotel in Paris for next weekend.",
        },
    )


@pytest.fixture
def patch_make_client(monkeypatch):
    fake = _canned_fake()
    monkeypatch.setattr("convgen.cli.make_client", lambda: fake)
    return fake


class TestBuildCommand:
    def test_build_writes_artifacts(self, toolbench_dir: Path, tmp_path: Path):
        out = tmp_path / "artifacts"
        result = runner.invoke(
            app,
            ["build", "--data-dir", str(toolbench_dir), "--out", str(out)],
        )
        assert result.exit_code == 0, result.stdout
        assert (out / "graph.pkl").exists()
        assert (out / "registry.pkl").exists()
        assert "Build summary" in result.stdout

    def test_build_missing_dir_errors(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "build",
                "--data-dir",
                str(tmp_path / "nope"),
                "--out",
                str(tmp_path / "a"),
            ],
        )
        assert result.exit_code != 0
        assert "does not exist" in result.stdout


class TestGenerateCommand:
    def test_generate_writes_jsonl(
        self, toolbench_dir: Path, tmp_path: Path, patch_make_client
    ):
        artifacts = tmp_path / "artifacts"
        runner.invoke(
            app,
            ["build", "--data-dir", str(toolbench_dir), "--out", str(artifacts)],
        )

        out = tmp_path / "outputs" / "data.jsonl"
        result = runner.invoke(
            app,
            [
                "generate",
                "--n",
                "3",
                "--seed",
                "1",
                "--out",
                str(out),
                "--artifacts",
                str(artifacts),
                "--no-cross-conversation-steering",
                "--clarification-rate",
                "0.0",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line)
            assert "conversation_id" in data
            assert "messages" in data
            assert "metadata" in data

    def test_generate_missing_artifacts_errors(self, tmp_path: Path):
        result = runner.invoke(
            app,
            [
                "generate",
                "--n",
                "1",
                "--artifacts",
                str(tmp_path / "no_artifacts"),
                "--out",
                str(tmp_path / "out.jsonl"),
            ],
        )
        assert result.exit_code != 0
        assert "graph" in result.stdout.lower()


class TestEvaluateCommand:
    def test_evaluate_prints_metrics(
        self, toolbench_dir: Path, tmp_path: Path, patch_make_client
    ):
        artifacts = tmp_path / "artifacts"
        runner.invoke(
            app,
            ["build", "--data-dir", str(toolbench_dir), "--out", str(artifacts)],
        )
        out = tmp_path / "out.jsonl"
        runner.invoke(
            app,
            [
                "generate",
                "--n",
                "3",
                "--seed",
                "1",
                "--out",
                str(out),
                "--artifacts",
                str(artifacts),
                "--no-cross-conversation-steering",
                "--clarification-rate",
                "0.0",
            ],
        )

        result = runner.invoke(
            app,
            ["evaluate", "--dataset", str(out), "--artifacts", str(artifacts)],
        )
        assert result.exit_code == 0, result.stdout
        assert "Unique tool coverage" in result.stdout
        assert "pair entropy" in result.stdout.lower()
        assert "Mean tool_correctness" in result.stdout

    def test_evaluate_missing_dataset_errors(self, tmp_path: Path):
        result = runner.invoke(
            app, ["evaluate", "--dataset", str(tmp_path / "nope.jsonl")]
        )
        assert result.exit_code != 0

