"""
Load `.env` for pytest so e2e tests see the same API keys as `convgen` CLI.

Without this, `tests/test_e2e.py` only reads `os.environ`; keys stored solely in
`.env` would be invisible and e2e would skip. Mirrors `convgen.cli` load order.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Package tests live next to `src/`; repo root is parents[1] of this file.
_REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(override=False)
load_dotenv(_REPO_ROOT / ".env", override=True)
