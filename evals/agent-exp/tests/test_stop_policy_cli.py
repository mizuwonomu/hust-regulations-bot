"""Kiểm tra entrypoint STOP policy: help, lifecycle offline và mã exit đã quy ước"""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from artifacts import read_snapshot
from stop_policy_synthetic import (
    CORPUS_ANSWER_MARKER,
    CORPUS_GOLD_MARKER,
    CORPUS_LINK_MARKER,
    FakeRuntime,
    default_answers,
    reviewed_corpus,
    write_corpus,
)

from stop_policy_eval import cli as stop_cli
from stop_policy_eval.artifacts import (
    load_run,
    prepare_run,
    read_input_traces,
    read_manifest,
    summarize_run,
)
from stop_policy_eval.capture import read_capture_manifest
from stop_policy_eval.cases import read_stop_cases
from stop_policy_eval.contracts import BASELINE_VARIANT
from stop_policy_eval.execution import request_fingerprint
from stop_policy_eval.schedule import build_stop_plan

AGENT_EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_EXP_ROOT.parents[1]
CLI_PATH = AGENT_EXP_ROOT / "scripts" / "stop_policy_eval" / "cli.py"

COMMANDS = (
    "capture-corpus",
    "inspect-capture",
    "import-later-hop",
    "export-later-hop",
    "validate-cases",
    "prepare",
    "prepare-hop0",
    "run",
    "summarize",
)
# Renderer prompt production có thể import dependency nặng
# Probe vẫn phải xác nhận không nạp credential, model loader hay database client
FORBIDDEN_MODULES = (
    "sentence_transformers",
    "FlagEmbedding",
    "langchain_groq",
    "langchain_openai",
    "src.rag.embedding_utils",
    "src.rag.reranker_utils",
    "src.rag.qa_chain",
    "src.rag.agent.llm_client",
    "src.rag.agent.gate",
    "src.rag.agent.loop",
    "src.rag.agent.tools",
    "src.ingestion.parser",
    "src.ingestion.ingest_regulations",
    "src.services.title_generator",
    "src.database",
    "chromadb",
    "langchain_chroma",
)
STRICT_IMPORT_BOUNDARY = (*FORBIDDEN_MODULES, "run_experiments", "dotenv", "dotenv.main")
OFFLINE_RUNTIME_BOUNDARY = (*FORBIDDEN_MODULES, "dotenv", "dotenv.main")
# Probe chặn mọi lời gọi load_dotenv và socket connect


class _BusyClient:
    """Client giả luôn thất bại transport, dùng cho nhánh execution failure"""

    model_name = "citation-agent"
    openai_api_base = "http://127.0.0.1:8080/v1"
    temperature = 0.0
    max_tokens = 4096
    request_timeout = 60.0
    max_retries = 2
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def invoke(self, prompt, extra_body=None):
        raise httpx.ConnectError("offline")


class _InterruptingClient(_BusyClient):
    def invoke(self, prompt, extra_body=None):
        raise KeyboardInterrupt()


def _direct(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        cwd=cwd or REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _import_probe(
    *args: str,
    forbidden_modules: tuple[str, ...] = FORBIDDEN_MODULES,
) -> tuple[int, list[str], bool, bool]:
    """Chạy command trong process mới và bắt import cùng lời gọi ra ngoài"""
    code = (
        "import builtins, json, runpy, socket, sys\n"
        "calls = []\n"
        "network_calls = []\n"
        "baseline_modules = set(sys.modules)\n"
        "def _deny_network(*positional, **keywords):\n"
        "    network_calls.append('socket.connect')\n"
        "    raise AssertionError('network calls are forbidden in the offline CLI probe')\n"
        "socket.socket.connect = _deny_network\n"
        "def _boom(*positional, **keywords):\n"
        "        calls.append('load_dotenv')\n"
        "        raise AssertionError('load_dotenv must never run in the offline CLI')\n"
        "original_import = builtins.__import__\n"
        "def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    module = original_import(name, globals, locals, fromlist, level)\n"
        "    if name == 'dotenv' or name.startswith('dotenv.'):\n"
        "        dotenv = sys.modules.get('dotenv')\n"
        "        dotenv_main = sys.modules.get('dotenv.main')\n"
        "        if dotenv is not None:\n"
        "            dotenv.load_dotenv = _boom\n"
        "        if dotenv_main is not None:\n"
        "            dotenv_main.load_dotenv = _boom\n"
        "    return module\n"
        "builtins.__import__ = _guarded_import\n"
        f"sys.argv = ['cli', *{list(args)!r}]\n"
        "code = 0\n"
        "try:\n"
        f"    runpy.run_path({str(CLI_PATH)!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    code = int(exc.code or 0)\n"
        f"forbidden = {list(forbidden_modules)!r}\n"
        "found = sorted(name for name in sys.modules if name in forbidden and name not in baseline_modules)\n"
        "print(json.dumps({'code': code, 'found': found, 'calls': len(calls), "
        "'network_calls': len(network_calls)}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return (
        payload["code"],
        payload["found"],
        bool(payload["calls"]),
        bool(payload["network_calls"]),
    )


def _fresh_process_probe(
    body: str,
    *,
    forbidden_modules: tuple[str, ...] = OFFLINE_RUNTIME_BOUNDARY,
) -> tuple[subprocess.CompletedProcess, dict]:
    """Chạy bootstrap test riêng trong process mới với dotenv và socket bị chặn"""
    guard = (
        "import builtins, json, socket, sys\n"
        "calls = []\n"
        "network_calls = []\n"
        "baseline_modules = set(sys.modules)\n"
        "def _deny_network(*positional, **keywords):\n"
        "    network_calls.append('socket.connect')\n"
        "    raise AssertionError('network calls are forbidden in the offline CLI probe')\n"
        "socket.socket.connect = _deny_network\n"
        "def _boom(*positional, **keywords):\n"
        "    calls.append('load_dotenv')\n"
        "    raise AssertionError('load_dotenv must never run in the offline CLI')\n"
        "original_import = builtins.__import__\n"
        "def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    module = original_import(name, globals, locals, fromlist, level)\n"
        "    if name == 'dotenv' or name.startswith('dotenv.'):\n"
        "        dotenv = sys.modules.get('dotenv')\n"
        "        dotenv_main = sys.modules.get('dotenv.main')\n"
        "        if dotenv is not None:\n"
        "            dotenv.load_dotenv = _boom\n"
        "        if dotenv_main is not None:\n"
        "            dotenv_main.load_dotenv = _boom\n"
        "    return module\n"
        "builtins.__import__ = _guarded_import\n"
        f"forbidden = {list(forbidden_modules)!r}\n"
        "result = {}\n"
    )
    trailer = (
        "found = sorted(name for name in sys.modules if name in forbidden and name not in baseline_modules)\n"
        "print(json.dumps({'result': result, 'found': found, 'dotenv_calls': len(calls), "
        "'network_calls': len(network_calls)}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", guard + body + trailer],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return completed, payload


def _assert_manifest_source_hashes_are_current(hashes: dict[str, str]) -> None:
    scripts_root = AGENT_EXP_ROOT / "scripts"
    for relative, digest in hashes.items():
        source = scripts_root / relative
        if not source.is_file():
            source = REPO_ROOT / relative
        assert source.is_file(), relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, relative


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory):
    return reviewed_corpus(tmp_path_factory.mktemp("cli"), name="cli")


@pytest.fixture(scope="module")
def prepared_paths(reviewed, tmp_path_factory):
    """Đường dẫn artifact dùng chung cho các probe subprocess"""
    root = tmp_path_factory.mktemp("cli-artifacts")
    run_dir = root / "run"
    prepare_run(
        build_stop_plan(
            list(reviewed.cases),
            reviewed.snapshot,
            reviewed.corpus,
            cases_path=reviewed.reviewed_path,
            snapshot_path=reviewed.seeds_path,
            split="dev",
            policies=["first"],
            repeats=1,
            variant=BASELINE_VARIANT,
        ),
        run_dir,
    )
    corpus_path, inventory_path = write_corpus(root, name="cli-corpus")
    return {
        "run_dir": run_dir,
        "corpus": corpus_path,
        "inventory": inventory_path,
        "seeds": reviewed.seeds_path,
        "cases": reviewed.reviewed_path,
        "capture_dir": reviewed.fixture.capture_dir,
        "states": root / "states.json",
    }


def test_direct_entrypoint_help_works_for_every_command(tmp_path):
    top = _direct("--help")
    assert top.returncode == 0, top.stderr
    for command in COMMANDS:
        completed = _direct(command, "--help", cwd=tmp_path)
        assert completed.returncode == 0, (command, completed.stderr)
        assert command in top.stdout


def test_direct_entrypoint_rejects_unknown_commands_and_flags(tmp_path):
    assert _direct("unknown-command", cwd=tmp_path).returncode == 2
    assert _direct("prepare", "--nope", cwd=tmp_path).returncode == 2
    unknown_variant = _direct(
        "prepare",
        "--seeds",
        "x",
        "--corpus",
        "y",
        "--cases",
        "z",
        "--split",
        "dev",
        "--policies",
        "first",
        "--repeats",
        "1",
        "--variant",
        "zero-shot",
        "--output-dir",
        "out",
        cwd=tmp_path,
    )
    assert unknown_variant.returncode == 2
    assert "invalid choice" in unknown_variant.stderr


def test_offline_commands_never_load_credentials_or_model_loaders(prepared_paths, tmp_path):
    code, found, dotenv_calls, network_calls = _import_probe(
        "--help", forbidden_modules=STRICT_IMPORT_BOUNDARY
    )
    assert (code, found, dotenv_calls, network_calls) == (0, [], False, False)
    code, found, dotenv_calls, network_calls = _import_probe(
        "inspect-capture",
        "--capture-dir",
        str(prepared_paths["capture_dir"]),
        forbidden_modules=STRICT_IMPORT_BOUNDARY,
    )
    assert (code, found, dotenv_calls, network_calls) == (0, [], False, False)
    code, found, dotenv_calls, network_calls = _import_probe(
        "validate-cases",
        "--seeds",
        str(prepared_paths["seeds"]),
        "--corpus",
        str(prepared_paths["corpus"]),
        "--cases",
        str(prepared_paths["cases"]),
        forbidden_modules=STRICT_IMPORT_BOUNDARY,
    )
    assert (code, found, dotenv_calls, network_calls) == (0, [], False, False)
    code, found, dotenv_calls, network_calls = _import_probe(
        "prepare",
        "--seeds",
        str(prepared_paths["seeds"]),
        "--corpus",
        str(prepared_paths["corpus"]),
        "--cases",
        str(prepared_paths["cases"]),
        "--variant",
        BASELINE_VARIANT,
        "--split",
        "heldout",
        "--policies",
        "first",
        "llm",
        "--repeats",
        "1",
        "--output-dir",
        str(tmp_path / "probe-run"),
    )
    assert (code, found, dotenv_calls, network_calls) == (0, [], False, False)
    code, found, dotenv_calls, network_calls = _import_probe(
        "summarize", "--run-dir", str(prepared_paths["run_dir"])
    )
    assert code in {0, 1}
    assert found == [] and dotenv_calls is False and network_calls is False


def test_capture_command_runs_with_a_fake_runtime(tmp_path, monkeypatch, capsys):
    corpus_path, inventory_path = write_corpus(tmp_path, name="cli-capture")
    runtime = FakeRuntime(default_answers())
    events = []
    env_file = tmp_path / "capture.env"

    def load_env_file(path):
        events.append(("env", Path(path)))

    def runtime_factory(settings):
        events.append(("runtime", None))
        return runtime

    monkeypatch.setattr(stop_cli, "_load_capture_env_file", load_env_file, raising=False)
    monkeypatch.setattr(stop_cli, "_live_capture_runtime_factory", runtime_factory)
    output_dir = tmp_path / "capture"
    code = stop_cli.main(
        [
            "capture-corpus",
            "--env-file",
            str(env_file),
            "--dataset",
            str(corpus_path),
            "--internal-dieu",
            str(inventory_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert code == 0
    assert events == [("env", env_file), ("runtime", None)]
    printed = capsys.readouterr().out
    for name in ("seeds.json", "cases_draft.jsonl", "cases_review.jsonl", "label_review.md"):
        assert str(output_dir / name) in printed
    assert len(runtime.questions) == 6

    inspected = stop_cli.main(["inspect-capture", "--capture-dir", str(output_dir)])
    assert inspected == 0
    assert "draft cases: 6" in capsys.readouterr().out

    refused = stop_cli.main(
        [
            "capture-corpus",
            "--dataset",
            str(corpus_path),
            "--internal-dieu",
            str(inventory_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert refused == 2
    assert events == [("env", env_file), ("runtime", None)]


def test_capture_env_loader_uses_selected_file_without_overriding_shell(
    tmp_path, monkeypatch
):
    import dotenv

    calls = []

    def load_dotenv(*, dotenv_path, override):
        calls.append((dotenv_path, override))

    monkeypatch.setattr(dotenv, "load_dotenv", load_dotenv)
    env_file = tmp_path / "capture.env"

    stop_cli._load_capture_env_file(env_file)

    assert calls == [(env_file, False)]


def test_capture_parser_defaults_env_file_and_accepts_an_override(tmp_path):
    command = [
        "capture-corpus",
        "--dataset",
        str(tmp_path / "corpus.json"),
        "--internal-dieu",
        str(tmp_path / "inventory.json"),
        "--output-dir",
        str(tmp_path / "capture"),
    ]

    defaults = stop_cli._build_parser().parse_args(command)
    override = tmp_path / "secrets.env"
    configured = stop_cli._build_parser().parse_args(
        [*command[:1], "--env-file", str(override), *command[1:]]
    )

    assert defaults.env_file == Path(".env")
    assert configured.env_file == override


def test_fresh_process_fake_capture_publishes_and_refuses_overwrite(tmp_path):
    corpus_path, inventory_path = write_corpus(tmp_path, name="fresh-capture")
    output_dir = tmp_path / "fresh-capture-bundle"
    env_file = tmp_path / "fresh-capture.env"
    body = f"""
sys.path.insert(0, {str(AGENT_EXP_ROOT / 'scripts')!r})
sys.path.insert(0, {str(AGENT_EXP_ROOT / 'tests')!r})
from stop_policy_synthetic import FakeRuntime, default_answers
from stop_policy_eval import cli
runtime = FakeRuntime(default_answers())
factory_calls = []
env_calls = []
def load_env_file(path):
    env_calls.append(str(path))
def factory(settings):
    factory_calls.append(True)
    return runtime
cli._load_capture_env_file = load_env_file
cli._live_capture_runtime_factory = factory
arguments = [
    "capture-corpus",
    "--env-file", {str(env_file)!r},
    "--dataset", {str(corpus_path)!r},
    "--internal-dieu", {str(inventory_path)!r},
    "--output-dir", {str(output_dir)!r},
]
first_code = cli.main(arguments)
second_code = cli.main(arguments)
result.update({{"codes": [first_code, second_code], "factory_calls": len(factory_calls), "questions": runtime.questions, "env_calls": env_calls}})
"""
    completed, payload = _fresh_process_probe(
        body,
        forbidden_modules=STRICT_IMPORT_BOUNDARY,
    )

    assert payload["result"] == {
        "codes": [0, 2],
        "factory_calls": 1,
        "questions": [
            row["user_input"] for row in json.loads(corpus_path.read_text(encoding="utf-8"))
        ],
        "env_calls": [str(env_file)],
    }
    assert payload["found"] == []
    assert payload["dotenv_calls"] == 0
    assert payload["network_calls"] == 0
    for name in ("seeds.json", "cases_draft.jsonl", "cases_review.jsonl", "label_review.md"):
        assert str(output_dir / name) in completed.stdout

    manifest = read_capture_manifest(output_dir)
    snapshot = read_snapshot(output_dir / "seeds.json")
    assert manifest.row_count == 6
    assert manifest.snapshot_id == snapshot.snapshot_id
    assert (output_dir / "cases_draft.jsonl").read_bytes() == (
        output_dir / "cases_review.jsonl"
    ).read_bytes()
    adapter = REPO_ROOT / "evals" / "common" / "single_pass_retrieval.py"
    assert manifest.captured_source_hashes["evals/common/single_pass_retrieval.py"] == (
        hashlib.sha256(adapter.read_bytes()).hexdigest()
    )
    assert {path.name for path in output_dir.iterdir()} == {
        "capture_manifest.json",
        "seeds.json",
        "cases_draft.jsonl",
        "cases_review.jsonl",
        "label_review.md",
    }


def test_fresh_process_first_lifecycle_reloads_exact_offline_artifacts(reviewed, tmp_path):
    run_dir = tmp_path / "fresh-first-run"
    body = f"""
sys.path.insert(0, {str(AGENT_EXP_ROOT / 'scripts')!r})
from stop_policy_eval import cli
shared = [
    "--seeds", {str(reviewed.seeds_path)!r},
    "--corpus", {str(reviewed.fixture.corpus_path)!r},
    "--cases", {str(reviewed.reviewed_path)!r},
]
commands = [
    ["validate-cases", *shared],
    ["prepare", *shared, "--variant", "baseline", "--split", "dev", "--policies", "first", "--repeats", "1", "--output-dir", {str(run_dir)!r}],
    ["run", "--run-dir", {str(run_dir)!r}],
    ["summarize", "--run-dir", {str(run_dir)!r}],
]
result.update({{"codes": [cli.main(command) for command in commands]}})
"""
    _, payload = _fresh_process_probe(body)

    assert payload["result"]["codes"] == [0, 0, 0, 0]
    assert payload["found"] == []
    assert payload["dotenv_calls"] == 0
    assert payload["network_calls"] == 0

    manifest, cases, traces, results = load_run(run_dir)
    assert manifest.status == "complete"
    assert manifest.execution_config is None
    assert manifest.policy_config["first"]["model_calls"] == 0
    assert len(traces) == len(manifest.input_trace_ids)
    assert len(results) == manifest.expected_policy_result_count
    _assert_manifest_source_hashes_are_current(manifest.executed_module_hashes)

    trace_text = json.dumps([trace.model_dump(mode="json") for trace in traces], ensure_ascii=False)
    for marker in (CORPUS_ANSWER_MARKER, CORPUS_GOLD_MARKER, CORPUS_LINK_MARKER):
        assert marker not in trace_text
    for case in cases:
        if case.label_reason is not None:
            assert case.label_reason not in trace_text
    for label_field in (
        "expected_action",
        "acceptable_dieu",
        "label_reason",
        "label_status",
        "semantic_group_id",
        "split",
        "fewshot_overlap",
        "overlap_notes",
    ):
        assert f'"{label_field}"' not in trace_text

    summary_before = (run_dir / "summary.json").read_bytes()
    assert summarize_run(run_dir) == "complete"
    assert (run_dir / "summary.json").read_bytes() == summary_before


def test_fresh_process_llm_lifecycle_records_config_and_reloads_requests(reviewed, tmp_path):
    run_dir = tmp_path / "fresh-llm-run"
    body = f"""
sys.path.insert(0, {str(AGENT_EXP_ROOT / 'scripts')!r})
from types import SimpleNamespace
from stop_policy_eval import artifacts, cli
class FakeClient:
    model_name = "fake-citation-agent"
    openai_api_base = "http://fake.invalid/v1"
    temperature = 0.0
    max_tokens = 4096
    request_timeout = 60.0
    max_retries = 2
    extra_body = {{"chat_template_kwargs": {{"enable_thinking": False}}}}
    api_key = "must-not-be-recorded"
    def __init__(self):
        self.calls = 0
    def invoke(self, prompt, extra_body=None):
        self.calls += 1
        return SimpleNamespace(
            content='{{"stop": true, "dieu": null}}',
            usage_metadata={{"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}},
        )
client = FakeClient()
factory_calls = []
def factory():
    factory_calls.append(True)
    return client
artifacts._default_client_factory = factory
shared = [
    "--seeds", {str(reviewed.seeds_path)!r},
    "--corpus", {str(reviewed.fixture.corpus_path)!r},
    "--cases", {str(reviewed.reviewed_path)!r},
]
commands = [
    ["validate-cases", *shared],
    ["prepare", *shared, "--variant", "baseline", "--split", "dev", "--policies", "llm", "--repeats", "1", "--output-dir", {str(run_dir)!r}],
    ["run", "--run-dir", {str(run_dir)!r}],
    ["summarize", "--run-dir", {str(run_dir)!r}],
]
result.update({{"codes": [cli.main(command) for command in commands], "factory_calls": len(factory_calls), "client_calls": client.calls}})
"""
    _, payload = _fresh_process_probe(body)

    assert payload["result"]["codes"] == [0, 0, 0, 0]
    assert payload["result"]["factory_calls"] == 1
    assert payload["result"]["client_calls"] > 0
    assert payload["found"] == []
    assert payload["dotenv_calls"] == 0
    assert payload["network_calls"] == 0

    manifest, cases, traces, results = load_run(run_dir)
    assert manifest.status == "complete"
    assert manifest.execution_config is not None
    assert manifest.execution_config.model_alias == "fake-citation-agent"
    assert manifest.execution_config.base_url == "http://fake.invalid/v1"
    assert manifest.policy_config["llm"]["status"] == "initialized"
    assert manifest.policy_config["llm"]["model_alias"] == "fake-citation-agent"
    assert manifest.execution_config_reason is None
    assert len(results) == manifest.expected_policy_result_count
    assert payload["result"]["client_calls"] == manifest.expected_trial_count
    assert all(record.outcome.status == "ok" for record in results)
    _assert_manifest_source_hashes_are_current(manifest.executed_module_hashes)

    manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert "must-not-be-recorded" not in manifest_text
    trace_text = json.dumps([trace.model_dump(mode="json") for trace in traces], ensure_ascii=False)
    for marker in (CORPUS_ANSWER_MARKER, CORPUS_GOLD_MARKER, CORPUS_LINK_MARKER):
        assert marker not in trace_text
    for case in cases:
        if case.label_reason is not None:
            assert case.label_reason not in trace_text
    for label_field in (
        "expected_action",
        "acceptable_dieu",
        "label_reason",
        "label_status",
        "semantic_group_id",
        "split",
        "fewshot_overlap",
        "overlap_notes",
    ):
        assert f'"{label_field}"' not in trace_text

    trace_by_id = {trace.input_trace_id: trace for trace in traces}
    assert all(
        record.request_fingerprint
        == request_fingerprint(
            trace_by_id[record.input_trace_id], "llm", manifest.execution_config
        )
        for record in results
    )
    summary_before = (run_dir / "summary.json").read_bytes()
    manifest_again, cases_again, traces_again, results_again = load_run(run_dir)
    assert (manifest_again, cases_again, traces_again, results_again) == (
        manifest,
        cases,
        traces,
        results,
    )
    assert summarize_run(run_dir) == "complete"
    assert (run_dir / "summary.json").read_bytes() == summary_before


def test_validate_prepare_first_lifecycle_and_summarize(reviewed, tmp_path, capsys):
    code = stop_cli.main(
        [
            "validate-cases",
            "--seeds",
            str(reviewed.seeds_path),
            "--corpus",
            str(reviewed.fixture.corpus_path),
            "--cases",
            str(reviewed.reviewed_path),
        ]
    )
    assert code == 0
    validated = capsys.readouterr().out
    assert "split dev: complete" in validated
    assert "eligible cases: 8" in validated

    run_dir = tmp_path / "cli-run"
    prepared = stop_cli.main(
        [
            "prepare",
            "--seeds",
            str(reviewed.seeds_path),
            "--corpus",
            str(reviewed.fixture.corpus_path),
            "--cases",
            str(reviewed.reviewed_path),
            "--variant",
            BASELINE_VARIANT,
            "--split",
            "dev",
            "--policies",
            "first",
            "--repeats",
            "2",
            "--output-dir",
            str(run_dir),
        ]
    )
    assert prepared == 0
    manifest = read_manifest(run_dir)
    assert manifest.status == "prepared"
    assert manifest.expected_trial_count == 8

    assert stop_cli.main(["prepare", "--seeds", str(reviewed.seeds_path), "--corpus", str(reviewed.fixture.corpus_path), "--cases", str(reviewed.reviewed_path), "--split", "dev", "--policies", "first", "--repeats", "2", "--output-dir", str(run_dir)]) == 2

    executed = stop_cli.main(["run", "--run-dir", str(run_dir)])
    assert executed == 0
    assert read_manifest(run_dir).status == "complete"

    assert stop_cli.main(["run", "--run-dir", str(run_dir)]) == 2
    assert stop_cli.main(["summarize", "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "summary.json").is_file()
    assert "Conclusion limits" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_run_reports_execution_failure_with_exit_code_one(reviewed, tmp_path, monkeypatch):
    import stop_policy_eval.artifacts as artifacts_module

    run_dir = tmp_path / "cli-failure"
    assert (
        stop_cli.main(
            [
                "prepare",
                "--seeds",
                str(reviewed.seeds_path),
                "--corpus",
                str(reviewed.fixture.corpus_path),
                "--cases",
                str(reviewed.reviewed_path),
                "--split",
                "dev",
                "--policies",
                "llm",
                "--repeats",
                "1",
                "--output-dir",
                str(run_dir),
            ]
        )
        == 0
    )
    monkeypatch.setattr(artifacts_module, "_default_client_factory", lambda: _BusyClient())
    assert stop_cli.main(["run", "--run-dir", str(run_dir)]) == 1
    manifest = read_manifest(run_dir)
    assert manifest.status == "completed_with_errors"
    assert manifest.execution_config is not None


def test_run_returns_130_on_interruption(reviewed, tmp_path, monkeypatch):
    import stop_policy_eval.artifacts as artifacts_module

    run_dir = tmp_path / "cli-interrupt"
    assert (
        stop_cli.main(
            [
                "prepare",
                "--seeds",
                str(reviewed.seeds_path),
                "--corpus",
                str(reviewed.fixture.corpus_path),
                "--cases",
                str(reviewed.reviewed_path),
                "--split",
                "dev",
                "--policies",
                "llm",
                "--repeats",
                "1",
                "--output-dir",
                str(run_dir),
            ]
        )
        == 0
    )
    monkeypatch.setattr(
        artifacts_module, "_default_client_factory", lambda: _InterruptingClient()
    )
    assert stop_cli.main(["run", "--run-dir", str(run_dir)]) == 130
    assert read_manifest(run_dir).status == "incomplete"
    assert stop_cli.main(["summarize", "--run-dir", str(run_dir)]) == 1


def test_data_errors_return_exit_code_two(reviewed, tmp_path):
    assert stop_cli.main(["summarize", "--run-dir", str(tmp_path / "missing")]) == 2
    assert (
        stop_cli.main(
            [
                "validate-cases",
                "--seeds",
                str(reviewed.seeds_path),
                "--corpus",
                str(tmp_path / "missing-corpus.json"),
                "--cases",
                str(reviewed.reviewed_path),
            ]
        )
        == 2
    )


def test_import_later_hop_refuses_an_existing_output(reviewed, tmp_path):
    base_path = reviewed.drafts_path
    output = tmp_path / "imported.jsonl"
    output.write_text("keep", encoding="utf-8")
    base_before = base_path.read_bytes()

    code = stop_cli.main(
        [
            "import-later-hop",
            "--seeds",
            str(reviewed.seeds_path),
            "--cases",
            str(base_path),
            "--states",
            str(tmp_path / "missing-states.jsonl"),
            "--source-run-dir",
            str(tmp_path / "missing-source-run"),
            "--output",
            str(output),
        ]
    )

    assert code == 2
    assert output.read_text(encoding="utf-8") == "keep"
    assert base_path.read_bytes() == base_before


def test_prepare_refuses_a_missing_requirement_class(reviewed, tmp_path, capsys):
    hop0_only = [case for case in reviewed.cases if case.source_hop == 0]
    cases_path = tmp_path / "hop0-only.jsonl"
    from stop_policy_eval.cases import write_new_stop_cases

    write_new_stop_cases(cases_path, hop0_only)
    code = stop_cli.main(
        [
            "prepare",
            "--seeds",
            str(reviewed.seeds_path),
            "--corpus",
            str(reviewed.fixture.corpus_path),
            "--cases",
            str(cases_path),
            "--split",
            "dev",
            "--policies",
            "first",
            "--repeats",
            "1",
            "--output-dir",
            str(tmp_path / "never"),
        ]
    )
    assert code == 2
    assert "post_follow" in capsys.readouterr().err
    assert not (tmp_path / "never").exists()


def test_validate_command_rejects_an_incomplete_hop0_roster(reviewed, tmp_path, capsys):
    hop0 = next(case for case in reviewed.cases if case.source_hop == 0)
    incomplete = [case for case in reviewed.cases if case.case_id != hop0.case_id]
    cases_path = tmp_path / "incomplete-review.jsonl"
    cases_path.write_text(
        "".join(
            json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for case in incomplete
        ),
        encoding="utf-8",
    )

    code = stop_cli.main(
        [
            "validate-cases",
            "--seeds",
            str(reviewed.seeds_path),
            "--corpus",
            str(reviewed.fixture.corpus_path),
            "--cases",
            str(cases_path),
        ]
    )

    assert code == 2
    assert "hop-0 roster" in capsys.readouterr().err
