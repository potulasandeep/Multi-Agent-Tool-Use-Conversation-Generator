# Sample diversity experiment output

**Note:** Example artifact from one completed run (`seed=42`, `n=100` per arm).
Regenerate on your machine with `python scripts/run_diversity_experiment.py`
after `convgen build`. Authoritative analysis and interpretation are in
[DESIGN.md §12](../DESIGN.md).

Generated with seed=42, requested n=100 per run. Row *Conversations* counts lines successfully written to each JSONL (failures are omitted).

| metric | Run A (no steering) | Run B (steering) | delta |
|---|---|---|---|
| Conversations | 100 | 100 | +0 |
| Unique tool coverage | 0.464 | 0.489 | +0.025 |
| Tool-pair entropy (bits) | 7.981 | 8.099 | +0.118 |
| Category Gini (lower=better) | 0.355 | 0.340 | -0.015 |
| Multi-step ratio (>=3) | 81.0% | 86.0% | +0.050 |
| Multi-tool ratio (>=2) | 79.0% | 80.0% | +0.010 |
| Clarification rate | 39.0% | 39.0% | +0.000 |
| Mean judge score | 3.44 | 3.44 | -0.007 |
|   tool_correctness | 3.16 | 3.15 | -0.010 |
|   grounding_fidelity | 2.57 | 2.55 | -0.020 |
|   naturalness | 4.20 | 4.15 | -0.050 |
|   task_completion | 3.84 | 3.89 | +0.050 |
