"""Offline tests cho helper nạp env runtime dùng chung."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals.common.runtime_env import DEFAULT_ENV_FILE, load_runtime_env

ENV_NAME = "HUST_RUNTIME_ENV_TEST"


def _write_env_file(tmp_path: Path, value: str) -> Path:
    path = tmp_path / "runtime.env"
    path.write_text(f"{ENV_NAME}={value}\n", encoding="utf-8")
    return path


def test_no_env_file_is_a_noop(monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    assert load_runtime_env(None) is None
    assert ENV_NAME not in os.environ
    # Không đoán `.env`, chỉ nạp khi được chỉ định
    assert DEFAULT_ENV_FILE == ".env"


def test_requested_env_file_is_loaded_without_overriding_process(monkeypatch, tmp_path):
    env_path = _write_env_file(tmp_path, "from-file")
    monkeypatch.setenv(ENV_NAME, "from-process")

    assert load_runtime_env(env_path) == env_path
    # override=False nên biến của process thắng biến trong file
    assert os.environ[ENV_NAME] == "from-process"

    monkeypatch.delenv(ENV_NAME)
    load_runtime_env(env_path)
    assert os.environ[ENV_NAME] == "from-file"


def test_override_true_replaces_process_value(monkeypatch, tmp_path):
    env_path = _write_env_file(tmp_path, "from-file")
    monkeypatch.setenv(ENV_NAME, "from-process")

    load_runtime_env(env_path, override=True)
    assert os.environ[ENV_NAME] == "from-file"


def test_missing_env_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_runtime_env(tmp_path / "missing.env")


def test_module_import_and_none_load_do_not_import_dotenv():
    # Process mới để module đã nạp từ pytest không che hành vi lazy
    code = (
        "import json, sys;"
        "sys.path.insert(0, '.');"
        "from evals.common.runtime_env import load_runtime_env;"
        "first = 'dotenv' in sys.modules;"
        "load_runtime_env(None);"
        "print(json.dumps({'before': first, 'after': 'dotenv' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[3]),
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report == {"before": False, "after": False}
