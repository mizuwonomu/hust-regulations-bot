"""Entrypoint repository cho harness prompt ablation bảo vệ luận án tiến sĩ."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    candidate for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (candidate / "pyproject.toml").is_file()
)
for path in (REPO_ROOT, SCRIPT_DIR):
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from diagnostic_subexp.doctoral_defense_prompt_ablation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
