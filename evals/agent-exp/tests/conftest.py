"""Isolate the agent experiment tests from database and dotenv fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

AGENT_EXP_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = AGENT_EXP_ROOT / "scripts"
REPO_ROOT = AGENT_EXP_ROOT.parents[1]
for path in (SCRIPTS_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
