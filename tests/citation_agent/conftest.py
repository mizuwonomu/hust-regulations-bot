"""Conftest riêng cho suite citation agent: offline, không DB, không .env, không tracing.

Suite này store-free nên phải cắt conftest gốc ở tests/ (nó load .env, import
src.api và định nghĩa Postgres fixtures) bằng cách chạy kèm
--confcutdir=tests/citation_agent từ repository root.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
