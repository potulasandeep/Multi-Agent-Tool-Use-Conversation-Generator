"""
Command-line interface for convgen.
"""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from convgen.agents.assistant import AssistantAgent
from convgen.agents.planner import PlannerAgent
from convgen.agents.user import UserAgent
from convgen.executor.mock import MockExecutor
from convgen.graph.builder import ToolGraph, build_tool_graph
from convgen.graph.sampler import (
    ConstrainedSampler,
    cli_generation_slot_constraints,
)
from convgen.io import read_dataset, write_conversation
from convgen.judge.judge import Judge
from convgen.judge.repair import RepairConfig, RepairLoop
from convgen.llm.client import LLMError, make_client
from convgen.orchestrator import Orchestrator
from convgen.registry.loader import load_registry
from convgen.steering.metrics import (
    category_gini,
    clarification_rate,
    mean_judge_scores,
    multi_step_ratio,
    multi_tool_ratio,
    tool_pair_entropy,
    unique_tool_coverage,
)
from convgen.steering.tracker import CoverageTracker

# Load cwd `.env` first, then repo-root `.env` with override so the project file
# always wins (avoids a stale cwd-only `.env` masking the real key in the repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(override=False)
load_dotenv(_REPO_ROOT / ".env", override=True)

app = typer.Typer(
    help="Offline synthetic multi-turn tool-use conversation generator."
)
console = Console()
logger = logging.getLogger("convgen.cli")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _die(msg: str) -> None:
    console.print(f"[red]error:[/red] {msg}")
    raise typer.Exit(code=1)


def _load_graph(artifacts_dir: Path) -> ToolGraph:
    graph_path = artifacts_dir / "graph.pkl"
    if not graph_path.exists():
        _die(
            f"No graph found at {graph_path}. "
            "Run `convgen build` first."
        )
    return ToolGraph.load(graph_path)


@app.command()
def build(
    data_dir: Path = typer.Option(
        Path("data/toolbench"),
        "--data-dir",
        help="Directory containing ToolBench JSON tool specs.",
    ),
    out_dir: Path = typer.Option(
        Path("artifacts"),
        "--out",
        help="Directory to write the built graph and registry.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Parse ToolBench data and build the tool graph."""
    _setup_logging(verbose)

    if not data_dir.exists():
        _die(f"Data directory does not exist: {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with console.status("[bold green]Loading registry..."):
        registry = load_registry(data_dir)
    if len(registry) == 0:
        _die(
            f"No usable tools found under {data_dir}. "
            "Make sure ToolBench JSON files are present."
        )

    with console.status("[bold green]Building tool graph..."):
        graph = build_tool_graph(registry)

    graph.save(out_dir / "graph.pkl")
    with (out_dir / "registry.pkl").open("wb") as f:
        pickle.dump(registry, f)

    counts = graph.edge_type_counts()
    table = Table(title="Build summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Tools", str(len(registry.tools)))
    table.add_row("Endpoints", str(len(registry)))
    table.add_row("Categories", str(len(registry.categories)))
    table.add_row("Graph edges (total)", str(graph.num_edges))
    for edge_type, count in counts.items():
        table.add_row(f"  {edge_type}", str(count))
    console.print(table)
    console.print(f"[green]OK[/green] Wrote artifacts to {out_dir}")


@app.command()
def generate(
    n: int = typer.Option(
        100, "--n", help="Number of conversations to generate."
    ),
    seed: int = typer.Option(
        42, "--seed", help="Random seed for reproducibility."
    ),
    out: Path = typer.Option(
        Path("outputs/dataset.jsonl"),
        "--out",
        help="Output JSONL path.",
    ),
    artifacts_dir: Path = typer.Option(
        Path("artifacts"),
        "--artifacts",
        help="Directory containing built artifacts.",
    ),
    no_cross_conversation_steering: bool = typer.Option(
        False,
        "--no-cross-conversation-steering",
        help=(
            "Disable cross-conversation diversity steering "
            "(Run A of the experiment)."
        ),
    ),
    clarification_rate_value: float = typer.Option(
        0.3,
        "--clarification-rate",
        help="Fraction of conversations that include a clarification turn.",
    ),
    min_chain: int = typer.Option(2, "--min-chain"),
    max_chain: int = typer.Option(5, "--max-chain"),
    quality_threshold: float = typer.Option(
        3.4,
        "--quality-threshold",
        help="Minimum mean judge score; below this triggers repair.",
    ),
    max_repairs: int = typer.Option(2, "--max-repairs"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate a dataset of conversations."""
    _setup_logging(verbose)

    graph = _load_graph(artifacts_dir)

    try:
        llm = make_client()
    except LLMError as e:
        _die(str(e))

    rng = random.Random(seed)
    tracker = (
        None if no_cross_conversation_steering else CoverageTracker()
    )
    sampler = ConstrainedSampler(graph, rng=rng, bias_provider=tracker)

    planner = PlannerAgent(llm)
    user_agent = UserAgent(llm)
    assistant = AssistantAgent(llm)
    executor = MockExecutor(seed=seed)
    judge = Judge(llm)
    orchestrator = Orchestrator(
        planner=planner,
        user=user_agent,
        assistant=assistant,
        executor=executor,
        clarification_rate=clarification_rate_value,
    )
    repair_loop = RepairLoop(
        orchestrator=orchestrator,
        judge=judge,
        config=RepairConfig(
            threshold=quality_threshold,
            max_repairs=max_repairs,
        ),
    )

    if out.exists():
        out.unlink()

    console.print(
        f"[cyan]Generating {n} conversations[/] "
        f"(steering={'OFF' if no_cross_conversation_steering else 'ON'}, "
        f"seed={seed})"
    )

    succeeded = 0
    failed = 0
    repaired = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating", total=n)
        for i in range(n):
            convo_id = f"conv_{i:05d}"
            convo_seed = seed * 100000 + i
            try:
                chain = sampler.sample(
                    cli_generation_slot_constraints(
                        i,
                        n,
                        min_length=min_chain,
                        max_length=max_chain,
                    )
                )
                conv = repair_loop.run(
                    chain=chain,
                    seed=convo_seed,
                    conversation_id=convo_id,
                )
                if conv.metadata.get("was_repaired", False):
                    repaired += 1
                if conv.metadata.get("failed"):
                    failed += 1
                else:
                    succeeded += 1
                if tracker is not None:
                    tracker.record(chain)
                write_conversation(out, conv)
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "Conversation %s crashed: %s", convo_id, e
                )
                failed += 1
            progress.update(task, advance=1)

    console.print()
    table = Table(title="Generation summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total requested", str(n))
    table.add_row("Succeeded", str(succeeded))
    table.add_row("Failed", str(failed))
    table.add_row("Repaired", str(repaired))
    table.add_row("Output", str(out))
    console.print(table)


@app.command()
def evaluate(
    dataset: Path = typer.Option(
        Path("outputs/dataset.jsonl"),
        "--dataset",
        help="Path to a JSONL dataset to evaluate.",
    ),
    artifacts_dir: Path = typer.Option(
        Path("artifacts"),
        "--artifacts",
        help="Artifacts directory (used to compute coverage fraction).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compute diversity and quality metrics over a generated dataset."""
    _setup_logging(verbose)

    if not dataset.exists():
        _die(f"Dataset not found: {dataset}")

    convs = list(read_dataset(dataset))
    if not convs:
        _die(f"Dataset {dataset} is empty.")

    total_tools: int | None = None
    registry_path = artifacts_dir / "registry.pkl"
    if registry_path.exists():
        with registry_path.open("rb") as f:
            registry = pickle.load(f)
        total_tools = len({t.name for t in registry.tools})

    coverage = unique_tool_coverage(convs, total_available_tools=total_tools)
    pair_ent = tool_pair_entropy(convs)
    gini = category_gini(convs)
    multi_step = multi_step_ratio(convs, min_steps=3)
    multi_tool = multi_tool_ratio(convs, min_tools=2)
    clar = clarification_rate(convs)
    quality = mean_judge_scores(convs)

    table = Table(title=f"Evaluation: {dataset.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Conversations", str(len(convs)))
    if total_tools:
        used = len(
            {t for c in convs for t in c.metadata.get("tools_used", [])}
        )
        table.add_row(
            "Unique tool coverage",
            f"{coverage:.3f} ({used}/{total_tools})",
        )
    else:
        table.add_row("Unique tool coverage", f"{coverage:.3f}")
    table.add_row("Tool-pair entropy (bits)", f"{pair_ent:.3f}")
    table.add_row("Category Gini (0=balanced)", f"{gini:.3f}")
    table.add_row("Multi-step ratio (≥3)", f"{multi_step:.1%}")
    table.add_row("Multi-tool ratio (≥2)", f"{multi_tool:.1%}")
    table.add_row("Clarification rate", f"{clar:.1%}")
    if quality:
        table.add_section()
        for key in (
            "tool_correctness",
            "grounding_fidelity",
            "naturalness",
            "task_completion",
            "mean",
        ):
            if key in quality:
                table.add_row(f"Mean {key}", f"{quality[key]:.2f}")
    else:
        table.add_section()
        table.add_row("Mean judge scores", "(no scored conversations)")
    console.print(table)


def main() -> Any:
    """Entry-point wrapper used by project scripts."""
    return app()


if __name__ == "__main__":
    app()

