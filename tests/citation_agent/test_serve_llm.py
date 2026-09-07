"""Shell test serve_llm.sh: subprocess với fake llama-server, PATH cô lập hoàn toàn.

PATH của subprocess CHỈ chứa các executable do fixture kiểm soát: fake
llama-server, python của interpreter đang chạy test (không resolve symlink để
giữ venv) và dirname (lệnh ngoài duy nhất script cần). Binary llama-server
thật không thể bị dò thấy ở bất kỳ case nào, kể cả case "không có server".
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.rag.config import CITATION_AGENT_BASE_URL, CITATION_AGENT_MODEL

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "src" / "rag" / "agent" / "scripts" / "serve_llm.sh"

FAKE_SERVER_BODY = '#!/bin/bash\nprintf "%s\\n" "$@" > "$FAKE_ARGFILE"\nexit 0\n'


def _make_controlled_bin(tmp_path, name, with_server):
    """Dựng bin dir chỉ chứa executable kiểm soát: dirname + python, và fake server nếu cần"""
    bin_dir = tmp_path / name
    bin_dir.mkdir()
    if with_server:
        argfile = tmp_path / f"{name}_argv.txt"
        fake_server = bin_dir / "llama-server"
        fake_server.write_text(FAKE_SERVER_BODY)
        fake_server.chmod(0o755)
    # dirname là lệnh ngoài duy nhất script dùng ngoài python; symlink có kiểm soát
    os.symlink(shutil.which("dirname"), bin_dir / "dirname")
    # Không resolve: giữ nguyên interpreter của môi trường test (venv)
    os.symlink(sys.executable, bin_dir / "python")
    return bin_dir, tmp_path / f"{name}_argv.txt"


@pytest.fixture
def isolated_env(tmp_path):
    """PATH chỉ gồm bin dir kiểm soát có fake server, không thư mục hệ thống nào khác"""
    home = tmp_path / "home"
    home.mkdir()
    bin_dir, argfile = _make_controlled_bin(tmp_path, "bin", with_server=True)
    env = {
        "PATH": str(bin_dir),
        "HOME": str(home),
        "FAKE_ARGFILE": str(argfile),
        "LLAMA_CPP_DIR": str(tmp_path / "llama.cpp"), # trống, chắc chắn không có server
        "LLAMA_CACHE": str(tmp_path / "cache"),
        "LANG": "C.UTF-8",
    }
    return {"env": env, "argfile": argfile, "tmp_path": tmp_path}


def run_script(env, *args, cwd):
    # bash gọi bằng đường dẫn tuyệt đối: execve không cần PATH, không thể dò nhầm binary
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def argv_of(isolated_env):
    return isolated_env["argfile"].read_text().splitlines()


class TestSyntax:
    def test_bash_n_passes(self):
        proc = subprocess.run(
            ["/bin/bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0


class TestInvocation:
    def test_resolves_repo_root_and_default_model_from_outside_repo(self, isolated_env, tmp_path):
        # Chạy từ thư mục ngoài repo: root phải suy từ vị trí script, không từ cwd
        proc = run_script(isolated_env["env"], cwd=tmp_path)
        assert proc.returncode == 0

        argv = argv_of(isolated_env)
        assert "-hf" in argv
        assert argv[argv.index("-hf") + 1] == CITATION_AGENT_MODEL

    def test_positional_model_overrides_configured_default(self, isolated_env, tmp_path):
        proc = run_script(isolated_env["env"], "unsloth/fake-model:Q8", cwd=tmp_path)
        assert proc.returncode == 0

        argv = argv_of(isolated_env)
        assert argv[argv.index("-hf") + 1] == "unsloth/fake-model:Q8"
        assert CITATION_AGENT_MODEL not in argv

    def test_path_server_selected_before_llama_cpp_dir(self, isolated_env):
        env = dict(isolated_env["env"])
        decoy_dir = Path(env["LLAMA_CPP_DIR"]) / "build" / "bin"
        decoy_dir.mkdir(parents=True)
        decoy_mark = isolated_env["tmp_path"] / "decoy_mark.txt"
        decoy = decoy_dir / "llama-server"
        decoy.write_text(f'#!/usr/bin/env bash\nprintf x > "{decoy_mark}"\n')
        decoy.chmod(0o755)

        proc = run_script(env, cwd=isolated_env["tmp_path"])
        assert proc.returncode == 0
        assert isolated_env["argfile"].exists() # fake trên PATH chạy
        assert not decoy_mark.exists() # candidate trong LLAMA_CPP_DIR bị bỏ qua

    def test_no_server_anywhere_fails_loudly(self, tmp_path):
        # Bin dir kiểm soát KHÔNG có fake server; PATH không còn chỗ nào khác
        # để dò được binary thật trên máy
        bin_dir, _ = _make_controlled_bin(tmp_path, "bin_noserver", with_server=False)
        home = tmp_path / "home"
        home.mkdir()
        env = {
            "PATH": str(bin_dir),
            "HOME": str(home),
            "LLAMA_CPP_DIR": str(tmp_path / "llama.cpp"), # trống
            "LLAMA_CACHE": str(tmp_path / "cache"),
            "LANG": "C.UTF-8",
        }
        proc = run_script(env, cwd=tmp_path)

        assert proc.returncode != 0
        assert "llama-server" in proc.stderr

    def test_env_overrides_reach_server_exactly_once(self, isolated_env, tmp_path):
        env = dict(isolated_env["env"])
        env["LLAMA_NGL"] = "33"
        env["LLAMA_CTX"] = "2048"
        env["LLAMA_PORT"] = "9099"

        proc = run_script(env, cwd=tmp_path)
        assert proc.returncode == 0

        argv = argv_of(isolated_env)
        assert argv.count("-ngl") == 1
        assert argv[argv.index("-ngl") + 1] == "33"
        assert argv.count("-c") == 1
        assert argv[argv.index("-c") + 1] == "2048"
        assert argv.count("--port") == 1
        assert argv[argv.index("--port") + 1] == "9099"

    def test_non_default_port_warns_about_mismatch(self, isolated_env, tmp_path):
        env = dict(isolated_env["env"])
        env["LLAMA_PORT"] = "9099"

        proc = run_script(env, cwd=tmp_path)
        assert proc.returncode == 0

        # Server vẫn nhận port được yêu cầu, warning chỉ nêu lệch base URL Python
        argv = argv_of(isolated_env)
        assert argv[argv.index("--port") + 1] == "9099"
        assert "CẢNH BÁO" in proc.stderr
        assert CITATION_AGENT_BASE_URL in proc.stderr

    def test_runtime_flags_preserved(self, isolated_env, tmp_path):
        proc = run_script(isolated_env["env"], cwd=tmp_path)
        assert proc.returncode == 0

        argv = argv_of(isolated_env)
        assert argv[argv.index("--host") + 1] == "127.0.0.1" # chỉ bind loopback
        assert "--jinja" in argv
        assert argv[argv.index("--reasoning") + 1] == "off"
        assert argv[argv.index("--parallel") + 1] == "1"
        assert argv[argv.index("--flash-attn") + 1] == "on"

    def test_alias_reaches_server(self, isolated_env, tmp_path):
        proc = run_script(isolated_env["env"], cwd=tmp_path)
        assert proc.returncode == 0

        argv = argv_of(isolated_env)
        assert argv[argv.index("--alias") + 1] == "citation-agent"
