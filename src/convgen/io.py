"""
JSONL reader/writer for conversation datasets.

Append-friendly writer so generation is crash-safe: each conversation
is flushed to disk as soon as it finishes. Reader is a generator so
large datasets don't have to fit in memory. The CLI materializes the
generator into a list at evaluate time because the metrics functions
need to iterate the corpus more than once.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from convgen.orchestrator import Conversation


def write_conversation(path: str | Path, conv: Conversation) -> None:
    """Append one conversation as a single JSONL line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = conv.model_dump_json()
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_dataset(path: str | Path, convs: list[Conversation]) -> None:
    """Write a full dataset, replacing the file if it exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for conv in convs:
            f.write(conv.model_dump_json() + "\n")


def read_dataset(path: str | Path) -> Iterator[Conversation]:
    """Yield Conversation objects from a JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield Conversation.model_validate(data)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(
                    f"Malformed line {i} in {p}: {e}"
                ) from e

