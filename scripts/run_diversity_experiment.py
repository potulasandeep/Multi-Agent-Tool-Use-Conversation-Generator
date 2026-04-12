"""
Diversity experiment runner.

Runs generation twice with the same seed:
  - Run A: steering disabled
  - Run B: steering enabled

Then computes quality and diversity metrics and writes a markdown table.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

from convgen.io import read_dataset
from convgen.steering.metrics import (
    category_gini,
    clarification_rate,
    mean_judge_scores,
    multi_step_ratio,
    multi_tool_ratio,
    tool_pair_entropy,
    unique_tool_coverage,
)

SEED = 42


def _run_cli(args: list[str]) -> None:
    print(f"$ convgen {' '.join(args)}")
    result = subprocess.run([sys.executable, "-m", "convgen.cli", *args], check=False)
    if result.returncode != 0:
        result = subprocess.run(["convgen", *args], check=False)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: convgen {' '.join(args)}")


def _load_total_tools(artifacts_dir: Path) -> int | None:
    p = artifacts_dir / "registry.pkl"
    if not p.exists():
        return None
    with p.open("rb") as f:
        registry = pickle.load(f)
    return len({t.name for t in registry.tools})


def _summarize(name: str, path: Path, total_tools: int | None) -> dict:
    convs = list(read_dataset(path))
    quality = mean_judge_scores(convs)
    return {
        "name": name,
        "n": len(convs),
        "coverage": unique_tool_coverage(convs, total_tools),
        "pair_entropy": tool_pair_entropy(convs),
        "gini": category_gini(convs),
        "multi_step": multi_step_ratio(convs, min_steps=3),
        "multi_tool": multi_tool_ratio(convs, min_tools=2),
        "clarification": clarification_rate(convs),
        "quality_mean": quality.get("mean", 0.0),
        "tool_correctness": quality.get("tool_correctness", 0.0),
        "grounding": quality.get("grounding_fidelity", 0.0),
        "naturalness": quality.get("naturalness", 0.0),
        "task_completion": quality.get("task_completion", 0.0),
    }


def _format_table(rows: list[dict]) -> str:
    headers = ["metric", "Run A (no steering)", "Run B (steering)", "delta"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * 4) + "|"]
    metrics = [
        ("Conversations", "n", "{:d}"),
        ("Unique tool coverage", "coverage", "{:.3f}"),
        ("Tool-pair entropy (bits)", "pair_entropy", "{:.3f}"),
        ("Category Gini (lower=better)", "gini", "{:.3f}"),
        ("Multi-step ratio (>=3)", "multi_step", "{:.1%}"),
        ("Multi-tool ratio (>=2)", "multi_tool", "{:.1%}"),
        ("Clarification rate", "clarification", "{:.1%}"),
        ("Mean judge score", "quality_mean", "{:.2f}"),
        ("  tool_correctness", "tool_correctness", "{:.2f}"),
        ("  grounding_fidelity", "grounding", "{:.2f}"),
        ("  naturalness", "naturalness", "{:.2f}"),
        ("  task_completion", "task_completion", "{:.2f}"),
    ]
    a, b = rows
    for label, key, fmt in metrics:
        va = a[key]
        vb = b[key]
        if isinstance(va, int):
            delta = f"{vb - va:+d}"
        else:
            d = vb - va
            delta = f"{d:+.3f}" if abs(d) < 1 else f"{d:+.2f}"
        lines.append(f"| {label} | {fmt.format(va)} | {fmt.format(vb)} | {delta} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if not (args.artifacts / "graph.pkl").exists():
        raise SystemExit(f"No graph at {args.artifacts / 'graph.pkl'}. Run `convgen build` first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_a_path = args.out_dir / "run_a_no_steering.jsonl"
    run_b_path = args.out_dir / "run_b_with_steering.jsonl"

    _run_cli(
        [
            "generate",
            "--n",
            str(args.n),
            "--seed",
            str(SEED),
            "--out",
            str(run_a_path),
            "--artifacts",
            str(args.artifacts),
            "--no-cross-conversation-steering",
        ]
    )
    _run_cli(
        [
            "generate",
            "--n",
            str(args.n),
            "--seed",
            str(SEED),
            "--out",
            str(run_b_path),
            "--artifacts",
            str(args.artifacts),
        ]
    )

    total_tools = _load_total_tools(args.artifacts)
    a_summary = _summarize("Run A (no steering)", run_a_path, total_tools)
    b_summary = _summarize("Run B (steering)", run_b_path, total_tools)
    if a_summary["n"] != b_summary["n"]:
        print(
            "\nWARNING: Run A wrote "
            f"{a_summary['n']} conversations to JSONL, Run B wrote "
            f"{b_summary['n']}. Metrics compare unequal samples "
            "(crashed generations are not recorded).\n"
        )
    table = _format_table([a_summary, b_summary])

    out_md = args.out_dir / "diversity_results.md"
    out_md.write_text(
        (
            "# Diversity experiment results\n\n"
            f"Generated with seed={SEED}, requested n={args.n} per run. "
            "Row *Conversations* counts lines successfully written to each "
            "JSONL (failures are omitted).\n\n"
            f"{table}\n"
        ),
        encoding="utf-8",
    )
    print("\n" + table + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

