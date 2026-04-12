"""Tests for the JSONL reader/writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from convgen.io import read_dataset, write_conversation, write_dataset
from convgen.orchestrator import Conversation


def _make(cid: str, intent: str = "test intent") -> Conversation:
    return Conversation(
        conversation_id=cid,
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        metadata={"user_intent": intent, "tools_used": ["x"]},
        judge_scores={"mean": 4.0},
    )


class TestWriteAndRead:
    def test_round_trip_single(self, tmp_path: Path):
        path = tmp_path / "out.jsonl"
        c = _make("c_0001")
        write_conversation(path, c)

        loaded = list(read_dataset(path))
        assert len(loaded) == 1
        assert loaded[0].conversation_id == "c_0001"
        assert loaded[0].metadata["user_intent"] == "test intent"
        assert loaded[0].judge_scores == {"mean": 4.0}

    def test_append_multiple(self, tmp_path: Path):
        path = tmp_path / "out.jsonl"
        write_conversation(path, _make("c_0001"))
        write_conversation(path, _make("c_0002"))
        write_conversation(path, _make("c_0003"))
        loaded = list(read_dataset(path))
        assert [c.conversation_id for c in loaded] == [
            "c_0001",
            "c_0002",
            "c_0003",
        ]

    def test_write_dataset_replaces_file(self, tmp_path: Path):
        path = tmp_path / "out.jsonl"
        write_conversation(path, _make("old"))
        write_dataset(path, [_make("new1"), _make("new2")])
        loaded = list(read_dataset(path))
        assert [c.conversation_id for c in loaded] == ["new1", "new2"]

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "deep" / "out.jsonl"
        write_conversation(path, _make("c_0001"))
        assert path.exists()


class TestReadErrors:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(read_dataset(tmp_path / "nope.jsonl"))

    def test_blank_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "out.jsonl"
        write_conversation(path, _make("c_0001"))
        with path.open("a", encoding="utf-8") as f:
            f.write("\n\n")
        write_conversation(path, _make("c_0002"))
        loaded = list(read_dataset(path))
        assert len(loaded) == 2

    def test_malformed_line_raises(self, tmp_path: Path):
        path = tmp_path / "out.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed line"):
            list(read_dataset(path))

