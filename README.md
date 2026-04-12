# convgen

**Pipeline:** ToolBench → Registry → Tool graph → Constrained sampler →
Multi-agent generator (planner / user / assistant) → **Mock executor** (offline
tool results + session state) → **LLM judge** → **Repair loop** → JSONL dataset.

All tool execution is fully offline (mocked), ensuring reproducibility and no
external **tool** API dependency (only the configured LLM provider is contacted:
planner, user, assistant, judge, and repair).

**Note:** `convgen generate` and the optional **`pytest -m e2e`** suite require an
**API key** (OpenAI or Anthropic). **`convgen build`**, **`convgen evaluate`**,
**`pytest -m "not e2e"`**, and the static files under **`examples/`** need **no**
network access.

Offline synthetic multi-turn **tool-use conversation generator** for ToolBench-style
API specs: builds a tool graph, samples constrained chains, runs a multi-agent
pipeline (planner / user / assistant) with a **mock executor** and **SessionStore**
for grounding, then **LLM-as-judge** scoring and **repair**. Optional
**cross-conversation steering** biases the sampler toward underused tools.

Architecture, tradeoffs, diversity experiment, and metadata schema are in
[DESIGN.md](./DESIGN.md) (weighted heavily in grading). JSONL record shape:
[DESIGN.md §12.4](./DESIGN.md#124-jsonl-record-schema).

## Deliverables vs this repo

| Requirement | Where |
|-------------|--------|
| Python package + pipeline | `src/convgen/` |
| CLI: `build`, `generate`, `evaluate` | `convgen` → `src/convgen/cli.py` |
| README (end-to-end) | This file |
| DESIGN (architecture, context, prompts, diversity) | [DESIGN.md](./DESIGN.md) |
| Unit tests | `tests/test_*.py` |
| Integration: repair loop | `tests/test_repair.py` |
| E2E: ≥100 samples, judge mean threshold | `tests/test_e2e.py` (`@pytest.mark.e2e`) |
| Steering off = Run A | `convgen generate --no-cross-conversation-steering` |
| Diversity script (same seed, A vs B) | `scripts/run_diversity_experiment.py` |
| Sample diversity table (from one run) | [examples/diversity_results.md](./examples/diversity_results.md) |
| Sample JSONL + experiment snapshot (shape / metrics) | [examples/sample_conversations.jsonl](./examples/sample_conversations.jsonl), [examples/sample_experiment_report.json](./examples/sample_experiment_report.json) |
| Minimal ToolBench-shaped fixture (smoke `build`) | `data/sample_toolbench/` |

**Steering implementation:** inverse-frequency **CoverageTracker** (not vector / mem0).
See DESIGN §10 and §14.2 for rationale and scaling limits. **Determinism:** no ANN
search in the shipped path; reproducibility strategy in DESIGN §7.3 and §13.2.

## Prerequisites

- **Python 3.10+**
- **ToolBench JSON** under `data/toolbench/` for full-scale runs (see
  [data/README.md](./data/README.md)). A **tiny** optional fixture ships under
  `data/sample_toolbench/` for instant `build` / short `generate` smoke tests.
- **API key** for **`convgen generate`** and **`pytest -m e2e`**: set
  `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` (copy from `.env.example`).
  (See the **Note** at the top for what runs offline.)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Three-command smoke test (no full ToolBench download)

Reviewers can sanity-check the pipeline end-to-end using only the bundled
fixture. **`convgen generate` requires an API key** (planner, user, assistant,
judge, and repair use the provider via `make_client()`). **`build`** and
**`evaluate`** are local only. **Unit tests** use `FakeLLMClient` and do not
require network access.

Internally, **`generate`** samples constrained tool chains from the graph, runs
the multi-agent loop, satisfies tool calls with the **mock executor** (no real
HTTP), scores each conversation with the **LLM judge**, and may run **repair**
when scores fall below the configured threshold before appending JSONL lines.

**Expected runtime:** about **30–60 seconds** for `--n 5`, depending on API
latency and whether repairs fire.

1. **Build** graph + registry from the tiny fixture:

   ```bash
   convgen build --data-dir data/sample_toolbench --out artifacts
   ```

2. **Generate** a handful of conversations (writes JSONL):

   ```bash
   convgen generate --n 5 --seed 42 --out outputs/smoke.jsonl --artifacts artifacts
   ```

3. **Evaluate** metrics on that file:

   ```bash
   convgen evaluate --dataset outputs/smoke.jsonl --artifacts artifacts
   ```

**Full ToolBench (or any compatible tree):** point `build` at your checkout with
`--data-dir` (default in the long workflow below is `data/toolbench`). The same
`artifacts/` directory is then used by `generate` / `evaluate` via
`--artifacts artifacts`.

**Judge:** this submission implements **LLM-as-judge** only for live scoring
(`Judge` + structured output). There is **no** separate heuristic-judge mode in
the CLI; **unit tests** use `FakeLLMClient` and do not call the network.

Example **record shape** (no API calls):
[examples/sample_conversations.jsonl](./examples/sample_conversations.jsonl).
Example **numeric experiment snapshot** (illustrative JSON):
[examples/sample_experiment_report.json](./examples/sample_experiment_report.json).

## End-to-end (full ToolBench layout)

**Expected runtime:** **`--n 100`** generation often takes **several minutes to
tens of minutes**, depending on API latency, how often the judge runs, and
repair attempts per conversation.

```bash
# 1) Ingest ToolBench and build graph + registry
convgen build --data-dir data/toolbench --out artifacts

# 2) Generate dataset (steering ON = Run B default)
convgen generate --n 100 --seed 42 --out outputs/dataset.jsonl

# 3) Same seed, steering OFF = Run A (diversity experiment control)
convgen generate --n 100 --seed 42 --no-cross-conversation-steering --out outputs/run_a.jsonl

# 4) Metrics + mean judge dimensions over JSONL
convgen evaluate --dataset outputs/dataset.jsonl --artifacts artifacts
```

`evaluate` loads each JSONL line with the `Conversation` Pydantic model (invalid
lines fail fast). It prints diversity metrics (coverage, pair entropy, category
Gini, multi-step / multi-tool / clarification rates) and aggregated judge means
when `judge_scores` are present.

### Useful `generate` flags

- `--no-cross-conversation-steering` — Run A (steering disabled)
- `--clarification-rate 0.3` — target fraction with clarification path
- `--min-chain 2 --max-chain 5` — chain length bounds for the sampler
- `--quality-threshold 3.4` — repair loop gate (mean judge score)
- `--max-repairs 2` — repair attempts per conversation

### Full diversity experiment (automates A/B, same seed)

After `convgen build`:

```bash
python scripts/run_diversity_experiment.py --n 100
```

Writes `outputs/diversity_results.md`. Numeric results and analysis are also in
DESIGN §12.

## Output format (JSONL)

Each line is one conversation: `conversation_id`, `messages` (role-tagged user /
assistant / tool, with `tool_calls` + `arguments` on assistant tool turns),
`metadata` (seed, chain, tools, repair flags, …), and `judge_scores` after
judging. See DESIGN section 12.4 and the assignment example (abbreviated) in the
handout.

## Tests

```bash
# Fast suite (no live LLM)
pytest -m "not e2e"

# Full suite including e2e (requires API key; ~100 conv × judge/repair cost).
# Keys in `.env` are loaded by `tests/conftest.py` (same idea as the CLI).
pytest -m e2e -v

ruff check src tests scripts
```

E2E threshold and justification: docstring in `tests/test_e2e.py` and DESIGN
section 12.3 / section 14.3.

## Project layout

```text
src/convgen/     # package (registry, graph, sampler, agents, executor, judge, CLI)
tests/           # unit, integration (repair), e2e
scripts/         # diversity experiment driver
data/toolbench/  # place ToolBench JSON here (see data/README.md)
artifacts/       # graph.pkl, registry.pkl after build
outputs/         # generated JSONL and experiment tables
examples/        # sample diversity_results.md, sample_conversations.jsonl, sample_experiment_report.json
scratch/         # e.g. failed_prompts.md referenced from DESIGN
data/sample_toolbench/  # tiny ToolBench-shaped fixture for smoke tests (optional)
```

## Packaging this submission (zip / upload)

**Do not include:** `.env`, real API keys, `.venv/`, `.llm_cache/`, full ToolBench
dumps, large `artifacts/*.pkl` or bulk `outputs/*.jsonl` unless the course
explicitly asks for them.

**Do include:** `src/`, `tests/` (including `tests/conftest.py`), `scripts/`,
`pyproject.toml`, `README.md`, `DESIGN.md`, `.env.example`, `.gitignore`,
`data/README.md`, `data/sample_toolbench/`, `examples/` (including sample JSONL /
JSON), empty-dir markers under `data/toolbench/.gitkeep`, `artifacts/.gitkeep`,
`outputs/.gitkeep`.

**Clean junk before zipping** (Unix/macOS, from submission root):

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .llm_cache .pytest_cache .ruff_cache
```

PowerShell (from submission root):

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -File -Filter *.pyc | Remove-Item -Force
Remove-Item -Recurse -Force .llm_cache, .pytest_cache, .ruff_cache -ErrorAction SilentlyContinue
```
