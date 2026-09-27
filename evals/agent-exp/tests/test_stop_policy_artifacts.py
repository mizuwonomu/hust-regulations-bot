"""Kiểm tra lifecycle artifact của STOP run: prepare, append, reload và finalize"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from stop_policy_synthetic import reviewed_corpus

from stop_policy_eval.artifacts import (
    StopArtifactError,
    execute_run,
    finalize_run,
    load_run,
    prepare_run,
    read_input_traces,
    read_manifest,
    summarize_run,
)
from stop_policy_eval.contracts import BASELINE_VARIANT
from stop_policy_eval.execution import request_fingerprint
from stop_policy_eval.schedule import build_stop_plan

PREPARED_FILES = {"input_traces.jsonl", "manifest.json", "report.md"}


class FakeGateClient:
    """Client gate giả ghi lại lời gọi và trả decision theo kịch bản"""

    def __init__(self, contents=None, *, error=None, interrupt_at=None):
        self.model_name = "citation-agent"
        self.openai_api_base = "http://127.0.0.1:8080/v1"
        self.temperature = 0.0
        self.max_tokens = 4096
        self.request_timeout = 60.0
        self.max_retries = 2
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.contents = list(contents) if contents else ['{"stop": true, "dieu": null}']
        self.error = error
        self.interrupt_at = interrupt_at
        self.calls = 0

    def invoke(self, prompt, extra_body=None):
        self.calls += 1
        if self.interrupt_at is not None and self.calls == self.interrupt_at:
            raise KeyboardInterrupt()
        if self.error is not None:
            raise self.error
        content = self.contents[min(self.calls - 1, len(self.contents) - 1)]
        return SimpleNamespace(
            content=content,
            usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        )


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory):
    """Review bundle tổng hợp dùng chung cho mọi test artifact"""
    return reviewed_corpus(tmp_path_factory.mktemp("artifacts"), name="artifacts")


def _plan(reviewed, *, split="dev", policies=("first", "llm"), repeats=2):
    return build_stop_plan(
        list(reviewed.cases),
        reviewed.snapshot,
        reviewed.corpus,
        cases_path=reviewed.reviewed_path,
        snapshot_path=reviewed.seeds_path,
        split=split,
        policies=list(policies),
        repeats=repeats,
        variant=BASELINE_VARIANT,
    )


def _prepare(reviewed, tmp_path: Path, *, policies=("first", "llm"), repeats=2, name="run"):
    return prepare_run(_plan(reviewed, policies=policies, repeats=repeats), tmp_path / name)


def _complete_first_run(reviewed, tmp_path: Path, name: str) -> Path:
    run_dir = tmp_path / name
    _prepare(reviewed, tmp_path, policies=("first",), repeats=1, name=name)
    execute_run(run_dir)
    assert finalize_run(run_dir) == "complete"
    return run_dir


def _manifest_payload(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _rewrite_manifest(run_dir: Path, payload: dict) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _result_lines(run_dir: Path, policy: str) -> list[dict]:
    path = run_dir / f"results_{policy}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_is_offline_and_refuses_an_existing_output(reviewed, tmp_path):
    run_dir = tmp_path / "offline"
    manifest = _prepare(reviewed, tmp_path, name="offline")

    assert manifest.status == "prepared"
    assert manifest.ended_at is None
    assert manifest.execution_config is None
    assert manifest.execution_config_reason
    assert manifest.policy_config["llm"] == {"status": "not_initialized"}
    assert manifest.executed_module_hashes
    assert "stop_policy_eval/schedule.py" in manifest.executed_module_hashes
    assert manifest.execution_revision
    assert manifest.expected_trial_count == len(_plan(reviewed).trials)
    assert manifest.expected_policy_result_count == manifest.expected_trial_count * 2
    assert {path.name for path in run_dir.iterdir() if path.is_file()} == PREPARED_FILES

    with pytest.raises(ValueError, match="destination already exists"):
        prepare_run(_plan(reviewed), run_dir)


def test_manifest_hashes_each_required_source_from_current_file_bytes(reviewed, tmp_path):
    manifest = _prepare(reviewed, tmp_path, policies=("first",), name="source-hashes")
    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    repo_root = Path(__file__).resolve().parents[3]
    required_sources = {
        "stop_policy_eval/cases.py": scripts_root / "stop_policy_eval" / "cases.py",
        "artifacts.py": scripts_root / "artifacts.py",
        "contracts.py": scripts_root / "contracts.py",
        "seed_cases.py": scripts_root / "seed_cases.py",
        "seed_capture.py": scripts_root / "seed_capture.py",
        "diagnostic_subexp/shared/execution.py": (
            scripts_root / "diagnostic_subexp" / "shared" / "execution.py"
        ),
        "diagnostic_subexp/shared/contracts.py": (
            scripts_root / "diagnostic_subexp" / "shared" / "contracts.py"
        ),
        "diagnostic_subexp/shared/provenance.py": (
            scripts_root / "diagnostic_subexp" / "shared" / "provenance.py"
        ),
        "diagnostic_subexp/shared/source_refs.py": (
            scripts_root / "diagnostic_subexp" / "shared" / "source_refs.py"
        ),
        "run_experiments.py": scripts_root / "run_experiments.py",
        "policies/first.py": scripts_root / "policies" / "first.py",
        "policies/llm.py": scripts_root / "policies" / "llm.py",
        "evals/common/single_pass_retrieval.py": (
            repo_root / "evals" / "common" / "single_pass_retrieval.py"
        ),
        "src/rag/agent/prompt.py": repo_root / "src" / "rag" / "agent" / "prompt.py",
        "src/rag/agent/gate.py": repo_root / "src" / "rag" / "agent" / "gate.py",
        "src/rag/agent/schema.py": repo_root / "src" / "rag" / "agent" / "schema.py",
        "src/rag/agent/llm_client.py": repo_root / "src" / "rag" / "agent" / "llm_client.py",
        "src/rag/config.py": repo_root / "src" / "rag" / "config.py",
        "src/rag/agent/loop.py": repo_root / "src" / "rag" / "agent" / "loop.py",
        "src/rag/agent/tools.py": repo_root / "src" / "rag" / "agent" / "tools.py",
        "src/ingestion/reference_parser.py": (
            repo_root / "src" / "ingestion" / "reference_parser.py"
        ),
    }

    for relative, path in required_sources.items():
        assert manifest.executed_module_hashes[relative] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    package_files = sorted((scripts_root / "stop_policy_eval").glob("*.py"))
    assert package_files
    assert all(
        path.relative_to(scripts_root).as_posix() in manifest.executed_module_hashes
        for path in package_files
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing executed module hashes"),
        ("changed", "executed module hash mismatch"),
        ("unknown", "unknown executed module paths"),
    ],
)
def test_reload_rejects_incomplete_or_untrusted_source_hashes(
    reviewed, tmp_path, monkeypatch, mutation, message
):
    import stop_policy_eval.artifacts as artifacts_module

    run_dir = tmp_path / f"hash-{mutation}"
    _complete_first_run(reviewed, tmp_path, run_dir.name)
    summary_before = (run_dir / "summary.json").read_bytes()
    payload = _manifest_payload(run_dir)
    if mutation == "missing":
        payload["executed_module_hashes"].pop("stop_policy_eval/schedule.py")
    elif mutation == "changed":
        payload["executed_module_hashes"]["src/rag/agent/prompt.py"] = "0" * 64
        monkeypatch.setattr(
            artifacts_module,
            "_verified_source",
            lambda *_: pytest.fail("source reload must follow module hash verification"),
        )
    else:
        payload["executed_module_hashes"]["src/rag/agent/not-a-module.py"] = "0" * 64
    _rewrite_manifest(run_dir, payload)

    with pytest.raises(StopArtifactError, match=message):
        load_run(run_dir)
    with pytest.raises(StopArtifactError, match=message):
        summarize_run(run_dir)

    assert (run_dir / "summary.json").read_bytes() == summary_before


def test_prepare_rejects_an_unresolved_dependency_before_output_creation(
    reviewed, tmp_path, monkeypatch
):
    import stop_policy_eval.artifacts as artifacts_module

    dependencies = getattr(artifacts_module, "DEPENDENCY_MODULES", ())
    monkeypatch.setattr(
        artifacts_module,
        "DEPENDENCY_MODULES",
        (*dependencies, "missing.stop_policy.local_dependency"),
        raising=False,
    )
    output_dir = tmp_path / "unresolved-source-run"

    with pytest.raises(StopArtifactError, match="required local dependency"):
        prepare_run(_plan(reviewed, policies=("first",)), output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("dependency.pyc", b"compiled", "does not resolve to Python source"),
        ("empty_dependency.py", b"", "source is empty"),
    ],
)
def test_source_module_resolution_rejects_non_source_and_empty_files(
    tmp_path, monkeypatch, filename, contents, message
):
    import stop_policy_eval.artifacts as artifacts_module

    module_name = f"_stop_policy_fixture_{filename.replace('.', '_')}"
    module_path = tmp_path / filename
    module_path.write_bytes(contents)
    monkeypatch.setattr(artifacts_module, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__file__=str(module_path)))

    with pytest.raises(StopArtifactError, match=message):
        artifacts_module._source_module_path(module_name)


def test_manifest_and_traces_exist_before_the_first_result(reviewed, tmp_path):
    run_dir = tmp_path / "prepared"
    manifest = _prepare(reviewed, tmp_path, policies=("first",), name="prepared")
    assert read_manifest(run_dir) == manifest
    traces = read_input_traces(run_dir)
    assert len(traces) == len({trial.input_trace_id for trial in manifest.trials})
    assert not (run_dir / "results_first.jsonl").exists()
    assert load_run(run_dir)[3] == []


def test_first_policy_creates_no_client_and_records_zero_model_calls(reviewed, tmp_path):
    run_dir = tmp_path / "first"
    _prepare(reviewed, tmp_path, policies=("first",), name="first")
    calls: list[str] = []

    def factory():
        calls.append("client")
        raise AssertionError("the first policy must not create a client")

    execute_run(run_dir, client_factory=factory)
    assert finalize_run(run_dir) == "complete"
    manifest, cases, traces, results = load_run(run_dir)
    assert calls == []
    assert manifest.execution_config is None
    assert len(results) == len(manifest.trials)
    assert {result.policy for result in results} == {"first"}
    assert all(result.outcome.status == "ok" for result in results)
    assert all(result.selected_position == 1 for result in results)
    assert manifest.policy_config["first"]["model_calls"] == 0


def test_llm_run_records_the_effective_config_and_binds_fingerprints(reviewed, tmp_path):
    run_dir = tmp_path / "llm"
    _prepare(reviewed, tmp_path, policies=("llm",), name="llm")
    client = FakeGateClient(contents=['{"stop": true, "dieu": null}'])

    execute_run(run_dir, client_factory=lambda: client)
    assert finalize_run(run_dir) == "complete"
    manifest, cases, traces, results = load_run(run_dir)
    assert client.calls == manifest.expected_trial_count
    assert manifest.execution_config is not None
    assert manifest.execution_config.model_alias == "citation-agent"
    assert manifest.execution_config.enable_thinking is False
    assert manifest.execution_config.max_completion_tokens == 4096
    assert manifest.execution_config_reason is None
    assert manifest.policy_config["llm"]["model_alias"] == "citation-agent"

    trace_by_id = {trace.input_trace_id: trace for trace in traces}
    for result in results:
        trace = trace_by_id[result.input_trace_id]
        assert result.request_fingerprint == request_fingerprint(
            trace, "llm", manifest.execution_config
        )
        assert result.request_fingerprint != trace.input_fingerprint
    with pytest.raises(ValueError, match="llm"):
        request_fingerprint(trace_by_id[results[0].input_trace_id], "llm", None)


def test_results_append_one_row_at_a_time(reviewed, tmp_path):
    run_dir = tmp_path / "append"
    _prepare(reviewed, tmp_path, policies=("first",), name="append")
    execute_run(run_dir)
    rows = _result_lines(run_dir, "first")
    assert [row["trial_id"] for row in rows] == [
        trial.trial_id for trial in read_manifest(run_dir).trials
    ]


@pytest.mark.parametrize(
    ("error", "content", "category"),
    [
        (httpx.ConnectError("offline"), None, "transport"),
        (
            openai.APITimeoutError(request=httpx.Request("POST", "http://127.0.0.1:8080/v1")),
            None,
            "timeout",
        ),
        (None, "not json", "schema"),
        (None, '{"stop": false, "dieu": 99}', "invalid_candidate"),
        (RuntimeError("boom"), None, "unexpected"),
    ],
)
def test_failures_become_error_outcomes_and_never_stop(reviewed, tmp_path, error, content, category):
    name = f"failure-{category}"
    run_dir = tmp_path / name
    _prepare(reviewed, tmp_path, policies=("llm",), repeats=1, name=name)
    client = FakeGateClient(contents=[content] if content else None, error=error)
    execute_run(run_dir, client_factory=lambda: client)
    assert finalize_run(run_dir) == "completed_with_errors"
    manifest, cases, traces, results = load_run(run_dir)
    assert len(results) == manifest.expected_trial_count
    for result in results:
        assert result.outcome.status == "error"
        assert result.outcome.decision is None
        assert result.outcome.error.category == category
        assert result.decision_correct is False
        assert result.selected_position is None


def test_interruption_preserves_completed_rows_and_marks_the_run_incomplete(reviewed, tmp_path):
    run_dir = tmp_path / "interrupt"
    _prepare(reviewed, tmp_path, policies=("llm",), name="interrupt")
    client = FakeGateClient(contents=['{"stop": true, "dieu": null}'], interrupt_at=2)
    with pytest.raises(KeyboardInterrupt):
        execute_run(run_dir, client_factory=lambda: client)
    assert len(_result_lines(run_dir, "llm")) == 1
    assert read_manifest(run_dir).status == "running"

    assert summarize_run(run_dir) == "incomplete"
    manifest, cases, traces, results = load_run(run_dir)
    assert manifest.status == "incomplete"
    assert manifest.ended_at is not None
    assert len(results) == 1


def test_rerunning_inference_on_a_finished_run_is_rejected_but_summarize_is_allowed(
    reviewed, tmp_path
):
    run_dir = tmp_path / "rerun"
    _prepare(reviewed, tmp_path, policies=("first",), name="rerun")
    execute_run(run_dir)
    assert finalize_run(run_dir) == "complete"
    with pytest.raises(StopArtifactError, match="result records"):
        execute_run(run_dir)
    assert summarize_run(run_dir) == "complete"
    assert summarize_run(run_dir) == "complete"


def test_deleting_debug_files_does_not_affect_reload_or_summary(reviewed, tmp_path):
    run_dir = tmp_path / "debug"
    _prepare(reviewed, tmp_path, policies=("first",), name="debug")
    execute_run(run_dir)
    assert summarize_run(run_dir) == "complete"
    summary_before = (run_dir / "summary.json").read_bytes()
    debug_dir = run_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    (debug_dir / "decisions_first.jsonl").write_text("{}\n", encoding="utf-8")
    (debug_dir / "run_first.log").write_text("noise\n", encoding="utf-8")
    assert summarize_run(run_dir) == "complete"
    assert (run_dir / "summary.json").read_bytes() == summary_before
    (debug_dir / "decisions_first.jsonl").unlink()
    assert load_run(run_dir)
    assert summarize_run(run_dir) == "complete"
    assert (run_dir / "summary.json").read_bytes() == summary_before


def test_no_result_local_sources_or_trajectories_are_created(reviewed, tmp_path):
    run_dir = tmp_path / "layout"
    _prepare(reviewed, tmp_path, policies=("first",), name="layout")
    execute_run(run_dir)
    assert finalize_run(run_dir) == "complete"
    names = {path.name for path in run_dir.iterdir() if path.is_file()}
    assert names == PREPARED_FILES | {"results_first.jsonl", "summary.json"}
    assert not any(name.startswith("trajectories") for name in names)
    assert not (run_dir / "sources").exists()
    assert not (run_dir / "cases.jsonl").exists()


def test_reload_detects_changed_source_bytes(reviewed, tmp_path):
    run_dir = tmp_path / "sources"
    _prepare(reviewed, tmp_path, policies=("first",), name="sources")
    execute_run(run_dir)
    seeds = Path(reviewed.seeds_path)
    original = seeds.read_bytes()
    try:
        seeds.write_bytes(original + b"\n")
        with pytest.raises((StopArtifactError, ValueError)):
            load_run(run_dir)
    finally:
        seeds.write_bytes(original)
    assert load_run(run_dir)


def test_reload_detects_a_different_cases_source(reviewed, tmp_path):
    run_dir = tmp_path / "cases-source"
    _prepare(reviewed, tmp_path, policies=("first",), name="cases-source")
    execute_run(run_dir)
    payload = _manifest_payload(run_dir)
    payload["cases_source"] = dict(payload["snapshot_source"])
    _rewrite_manifest(run_dir, payload)
    with pytest.raises(StopArtifactError):
        load_run(run_dir)


def test_reload_detects_changed_traces(reviewed, tmp_path):
    run_dir = tmp_path / "traces"
    _prepare(reviewed, tmp_path, policies=("first",), name="traces")
    execute_run(run_dir)
    lines = (run_dir / "input_traces.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["grammar_text"] = payload["grammar_text"] + " "
    payload["grammar_hash"] = hashlib.sha256(payload["grammar_text"].encode("utf-8")).hexdigest()
    lines[0] = json.dumps(payload, ensure_ascii=False)
    (run_dir / "input_traces.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(StopArtifactError):
        load_run(run_dir)


def test_reload_detects_a_changed_schedule(reviewed, tmp_path):
    run_dir = tmp_path / "schedule"
    _prepare(reviewed, tmp_path, policies=("first",), name="schedule")
    execute_run(run_dir)
    payload = _manifest_payload(run_dir)
    payload["trials"] = [payload["trials"][1], payload["trials"][0], *payload["trials"][2:]]
    _rewrite_manifest(run_dir, payload)
    with pytest.raises(StopArtifactError):
        load_run(run_dir)


def test_reload_detects_duplicate_and_unknown_results(reviewed, tmp_path):
    run_dir = tmp_path / "results"
    _prepare(reviewed, tmp_path, policies=("first",), name="results")
    execute_run(run_dir)
    path = run_dir / "results_first.jsonl"
    original = path.read_text(encoding="utf-8")
    first_line = original.splitlines()[0]

    path.write_text(original + first_line + "\n", encoding="utf-8")
    with pytest.raises(StopArtifactError):
        load_run(run_dir)

    bogus = json.loads(first_line)
    bogus["trial_id"] = "unknown-trial"
    path.write_text(original + json.dumps(bogus, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(StopArtifactError):
        load_run(run_dir)

    path.write_text(original, encoding="utf-8")
    assert load_run(run_dir)


def test_reload_detects_changed_request_fingerprints(reviewed, tmp_path):
    run_dir = tmp_path / "fingerprints"
    _prepare(reviewed, tmp_path, policies=("llm",), repeats=1, name="fingerprints")
    client = FakeGateClient(contents=['{"stop": true, "dieu": null}'])
    execute_run(run_dir, client_factory=lambda: client)
    assert load_run(run_dir)

    payload = _manifest_payload(run_dir)
    payload["execution_config"]["temperature"] = 0.9
    _rewrite_manifest(run_dir, payload)
    with pytest.raises(StopArtifactError, match="fingerprint"):
        load_run(run_dir)


@pytest.mark.parametrize("extra_name", ["results_llm.jsonl", "results_unregistered.jsonl"])
def test_summarize_rejects_unlisted_policy_result_files(reviewed, tmp_path, extra_name):
    run_dir = _complete_first_run(reviewed, tmp_path, extra_name.replace(".jsonl", ""))
    summary_before = (run_dir / "summary.json").read_bytes()
    (run_dir / extra_name).write_text("", encoding="utf-8")

    with pytest.raises(StopArtifactError, match="unlisted policy result file"):
        load_run(run_dir)
    with pytest.raises(StopArtifactError, match="unlisted policy result file"):
        summarize_run(run_dir)

    assert (run_dir / "summary.json").read_bytes() == summary_before


def test_reload_rejects_a_mislabeled_row_in_a_listed_policy_file(reviewed, tmp_path):
    run_dir = _complete_first_run(reviewed, tmp_path, "mislabeled-policy")
    path = run_dir / "results_first.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["policy"] = "llm"
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(StopArtifactError, match="has policy 'llm'"):
        load_run(run_dir)


def test_summarize_rejects_a_review_file_missing_a_hop0_case(reviewed, tmp_path):
    run_dir = _complete_first_run(reviewed, tmp_path, "missing-hop0")
    summary_before = (run_dir / "summary.json").read_bytes()
    cases_path = tmp_path / "missing-hop0-review.jsonl"
    rows = [
        json.loads(line)
        for line in reviewed.reviewed_path.read_text(encoding="utf-8").splitlines()
    ]
    removed = next(row for row in rows if row["source_hop"] == 0)
    rows.remove(removed)
    cases_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    payload = _manifest_payload(run_dir)
    payload["cases_source"]["path"] = str(cases_path)
    payload["cases_source"]["sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    _rewrite_manifest(run_dir, payload)

    with pytest.raises(StopArtifactError, match="hop-0 roster.*missing"):
        summarize_run(run_dir)

    assert (run_dir / "summary.json").read_bytes() == summary_before


def test_execution_provenance_treats_a_clean_tree_as_not_dirty(monkeypatch):
    import stop_policy_eval.artifacts as artifacts_module

    def fake_git(arguments):
        if arguments[:1] == ["status"]:
            return ""
        return "0" * 40 + "\n"

    monkeypatch.setattr(artifacts_module, "_git_output", fake_git)
    provenance = artifacts_module.execution_provenance()
    assert provenance["execution_dirty"] is False
    assert provenance["execution_revision"] == "0" * 40
    assert "stop_policy_eval/schedule.py" in provenance["executed_module_hashes"]


def test_reload_detects_stale_serialized_correctness(reviewed, tmp_path):
    run_dir = tmp_path / "correctness"
    _prepare(reviewed, tmp_path, policies=("first",), repeats=1, name="correctness")
    execute_run(run_dir)
    path = run_dir / "results_first.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["decision_correct"] = not tampered["decision_correct"]
    lines[0] = json.dumps(tampered, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(StopArtifactError):
        load_run(run_dir)
