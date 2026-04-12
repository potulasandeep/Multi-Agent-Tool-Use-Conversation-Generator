# ToolBench data directory

Place **ToolBench** (or compatible) tool JSON files here so `convgen build
--data-dir data/toolbench` can run.

## Expected layout

The loader accepts:

```text
data/toolbench/<Category>/<tool>.json
data/toolbench/<Category>/<subdir>/<tool>.json
```

The first path segment under `data/toolbench/` is treated as the **category**
name. Files placed directly under `data/toolbench/` use category `Unknown`.

Each JSON file should follow the usual ToolBench shape (tool metadata plus
`api_list` endpoints). Malformed fragments are **skipped** at parameter,
endpoint, or file level; see `src/convgen/registry/loader.py`.

## Shipping policy

This repository **does not** include ToolBench dumps (size and licensing).
Course staff and reviewers should point `--data-dir` at their own checkout.
For a quick sanity check, any sufficiently rich subset with multiple categories
and endpoints is enough for build + generate.

## Bundled minimal fixture (optional quick start)

A **tiny** ToolBench-shaped tree lives under `data/sample_toolbench/` (a few
endpoints across `Travel` and `Weather`). It is **not** a replacement for real
ToolBench; it exists so reviewers can run `convgen build --data-dir
data/sample_toolbench` immediately, then try a short `generate` run before
pointing at a full dump.
