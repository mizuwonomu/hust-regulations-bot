"""Kiểm tra entrypoint, lifecycle một lệnh và binding trial-trace của diagnostic"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

AGENT_EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_EXP_ROOT.parents[1]
sys.path.insert(0, str(AGENT_EXP_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from diagnostic_subexp.article_42_observation_block_order import cli as observation_cli
from diagnostic_subexp.article_42_observation_block_order.artifacts import (
    load_observation_run,
)
from diagnostic_subexp.candidate_pair_41_3_position import cli as run_diagnostics
from diagnostic_subexp.candidate_pair_41_3_position.artifacts import (
    DiagnosticArtifactError,
    load_candidate_run,
)
from run_fixed_diagnostics_ab import _prepare_suites

A_CLI_PATH = (
    AGENT_EXP_ROOT / "scripts" / "diagnostic_subexp" / "candidate_pair_41_3_position" / "cli.py"
)
B_CLI_PATH = (
    AGENT_EXP_ROOT / "scripts"
    / "diagnostic_subexp"
    / "article_42_observation_block_order"
    / "cli.py"
)


def _python() -> Path:
    """Chọn interpreter đang chạy test để gọi subprocess"""
    return Path(sys.executable)


def _cli(*args: str, script: Path = A_CLI_PATH) -> subprocess.CompletedProcess:
    """Chạy CLI trực tiếp từ worktree root"""
    return subprocess.run(
        [str(_python()), str(script), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _trace_payloads(run_dir: Path) -> list[dict]:
    """Đọc toàn bộ input trace đã lưu"""
    return [
        json.loads(line)
        for line in (run_dir / "input_traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _parts(outcome):
    """Trả run_dir, manifest, plan từ một PrepareOutcome."""
    return outcome.run_dir, outcome.manifest, outcome.plan


def _rewrite_manifest(run_dir: Path, payload: dict) -> None:
    """Ghi lại manifest với payload đã bị sửa đổi"""
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_direct_entrypoint_help_bootstraps_repository_imports():
    """Bắt entrypoint direct chạy khác môi trường pytest conftest"""
    for script in (A_CLI_PATH, B_CLI_PATH):
        completed = _cli("--help", script=script)
        assert completed.returncode == 0, completed.stderr
        assert "prepare" in completed.stdout


def test_direct_entrypoint_rejects_an_unknown_phase():
    """Bắt entrypoint direct trả exit code khác 2 khi argv sai"""
    completed = _cli("unknown-phase")
    assert completed.returncode == 2
    assert "invalid choice" in completed.stderr


@pytest.mark.parametrize(
    ("script", "expected_trials", "expected_results"),
    [(A_CLI_PATH, 30, 30), (B_CLI_PATH, 12, 12)],
)
def test_first_policy_lifecycle_completes_in_one_command(
    tmp_path: Path,
    script: Path,
    expected_trials: int,
    expected_results: int,
):
    """Bắt run first chạy rồi finalize trong một command và ghi đủ artifact"""
    run_dir = tmp_path / "run"

    prepared = _cli(
        "prepare",
        "--output-dir",
        str(run_dir),
        "--policies",
        "first",
        script=script,
    )
    assert prepared.returncode == 0, prepared.stderr
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "prepared"

    executed = _cli("run", "--run-dir", str(run_dir), script=script)
    assert executed.returncode == 0, executed.stderr

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["ended_at"] is not None
    assert len(manifest["trials"]) == expected_trials
    results = (run_dir / "results_first.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results) == expected_results
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "report.md").exists()
    assert not (run_dir / "results_llm.jsonl").exists()
    assert not (run_dir / "sources").exists()


def test_prepare_does_not_import_credentials_or_construct_a_client(tmp_path: Path, monkeypatch):
    """Bắt prepare dựng model client hoặc đọc cấu hình bí mật"""
    import policies.llm

    def _refuse(*_args, **_kwargs):
        raise AssertionError("prepare must not construct an LLM client")

    monkeypatch.setattr(policies.llm, "create_client", _refuse)
    run_dir, manifest, plan = _parts(run_diagnostics.prepare_run(None, tmp_path / "run", ["first"]))
    assert manifest.status == "prepared"
    assert plan.expected_policy_result_count == 30
    assert (run_dir / "manifest.json").exists()


def test_summarize_recomputes_offline_after_finalize(tmp_path: Path):
    """Bắt summarize không tái lập được summary từ artifact đã lưu"""
    run_dir = tmp_path / "run"
    assert _cli(
        "prepare",
        "--output-dir",
        str(run_dir),
        "--policies",
        "first",
        script=B_CLI_PATH,
    ).returncode == 0
    assert _cli("run", "--run-dir", str(run_dir), script=B_CLI_PATH).returncode == 0
    before = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    completed = _cli("summarize", "--run-dir", str(run_dir), script=B_CLI_PATH)
    assert completed.returncode == 0, completed.stderr
    after = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert after["scheduled"] == before["scheduled"] == 12
    assert after["valid"] == before["valid"] == 12


def test_reload_rejects_a_trial_pointing_at_another_valid_trace(tmp_path: Path):
    """Bắt manifest trỏ trial A0 sang trace A1 hợp lệ nhưng khác arm"""
    run_dir, _, _ = _parts(run_diagnostics.prepare_run(None, tmp_path / "run", ["first"]))
    traces = _trace_payloads(run_dir)
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    donor = next(trace for trace in traces if trace["arm_id"] == "a1-pair-23-3-first")
    index = next(
        index
        for index, trial in enumerate(payload["trials"])
        if trial["arm_id"] == "a0-original-control"
    )
    payload["trials"][index]["input_trace_id"] = donor["input_trace_id"]
    payload["trials"][index]["presented_candidates"] = donor["presented_candidates"]
    _rewrite_manifest(run_dir, payload)

    with pytest.raises(DiagnosticArtifactError, match="bound trace"):
        load_candidate_run(run_dir)


def test_reload_rejects_a_trial_with_a_changed_request_identity(tmp_path: Path):
    """Bắt manifest sửa một field request identity của trial"""
    run_dir, _, _ = _parts(run_diagnostics.prepare_run(None, tmp_path / "run", ["first"]))
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["trials"][0]["request_fingerprint"] = "0" * 64
    _rewrite_manifest(run_dir, payload)

    with pytest.raises(DiagnosticArtifactError, match="request_fingerprint"):
        load_candidate_run(run_dir)


def test_reload_rejects_a_trial_with_changed_treatment_fields(tmp_path: Path):
    """Bắt manifest sửa grammar hash hoặc hash message hiệu lực của trial"""
    run_dir, _, _ = _parts(run_diagnostics.prepare_run(None, tmp_path / "run", ["first"]))
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["trials"][0]["effective_messages_hash"] = "0" * 64
    _rewrite_manifest(run_dir, payload)

    with pytest.raises(DiagnosticArtifactError, match="effective_messages_hash"):
        load_candidate_run(run_dir)


def test_reload_rejects_a_missing_or_extra_trace(tmp_path: Path):
    """Bắt manifest trace set khác số trace đã lưu"""
    run_dir, _, _ = _parts(run_diagnostics.prepare_run(None, tmp_path / "run", ["first"]))
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["input_trace_ids"] = payload["input_trace_ids"] + ["extra-trace"]
    _rewrite_manifest(run_dir, payload)

    with pytest.raises(DiagnosticArtifactError):
        load_candidate_run(run_dir)


def test_reload_accepts_the_untampered_run(tmp_path: Path):
    """Bảo đảm reload vẫn chấp nhận artifact nguyên vẹn sau khi siết kiểm tra"""
    run_dir, manifest, plan = _parts(observation_cli.prepare_run(None, tmp_path / "run", ["first"]))
    _, cases, traces, results = load_observation_run(run_dir)
    observation_cli.execute_schedule(run_dir, manifest, cases, traces, {})
    assert observation_cli.finalize_run(run_dir) == "complete"

    reloaded_manifest, _, reloaded_traces, reloaded_results = load_observation_run(run_dir)
    assert len(reloaded_traces) == len(plan.traces)
    assert len(reloaded_results) == 12
    assert reloaded_manifest.status == "complete"


def test_manifest_contract_carries_the_execution_dirty_flag(tmp_path: Path, monkeypatch):
    """Fresh manifest ghi đúng dirty state của lần prepare."""
    from diagnostic_subexp.shared import provenance as shared_provenance

    monkeypatch.setattr(shared_provenance, "git_dirty", lambda: True)
    run_dir, manifest, _ = _parts(run_diagnostics.prepare_run(None, tmp_path / "dirty", ["first"]))
    assert manifest.execution_dirty is True
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["execution_dirty"] is True
    assert payload["execution_revision"] is not None

    monkeypatch.setattr(shared_provenance, "git_dirty", lambda: False)
    clean_dir, clean_manifest, _ = _parts(run_diagnostics.prepare_run(None, tmp_path / "clean", ["first"]))
    assert clean_manifest.execution_dirty is False
    assert json.loads((clean_dir / "manifest.json").read_text(encoding="utf-8"))[
        "execution_dirty"
    ] is False


def test_fresh_manifest_requires_execution_provenance(tmp_path: Path):
    """Fresh schema-v3 từ chối manifest thiếu hoặc rỗng execution provenance."""
    run_dir, _, _ = _parts(run_diagnostics.prepare_run(None, tmp_path / "run", ["first"]))
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    for field in ("execution_dirty", "execution_revision", "executed_module_hashes"):
        for broken_value in ("missing", None):
            broken = json.loads(json.dumps(payload))
            if broken_value == "missing":
                broken.pop(field)
            else:
                broken[field] = None
            (run_dir / "manifest.json").write_text(
                json.dumps(broken, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            with pytest.raises(DiagnosticArtifactError):
                load_candidate_run(run_dir)

    broken = json.loads(json.dumps(payload))
    broken["executed_module_hashes"] = {}
    (run_dir / "manifest.json").write_text(
        json.dumps(broken, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(DiagnosticArtifactError, match="executed module hashes"):
        load_candidate_run(run_dir)

    broken = json.loads(json.dumps(payload))
    first_module = next(iter(broken["executed_module_hashes"]))
    broken["executed_module_hashes"][first_module] = "not-a-digest"
    (run_dir / "manifest.json").write_text(
        json.dumps(broken, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(DiagnosticArtifactError, match="not a SHA-256 digest"):
        load_candidate_run(run_dir)


def test_fresh_manifest_records_real_package_relative_module_hashes(tmp_path: Path):
    """Mọi entry executed_module_hashes phải trỏ đúng file package-relative."""
    from diagnostic_subexp.shared.provenance import sha256_file
    from diagnostic_subexp.shared.source_refs import scripts_root

    run_dir, _, _ = _parts(run_diagnostics.prepare_run(None, tmp_path / "run", ["first"]))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest["executed_module_hashes"]
    assert hashes
    for relative, digest in hashes.items():
        assert (scripts_root() / relative).is_file()
        assert digest == sha256_file(scripts_root() / relative)


def test_prepare_many_validates_every_suite_before_writing(tmp_path: Path, monkeypatch):
    """Bắt prepare nhiều suite ghi suite đầu trước khi suite sau được validate"""

    def _fail_for_b(_spec_path, _policies):
        raise ValueError("injected B preflight failure")

    monkeypatch.setattr(observation_cli, "validate_run", _fail_for_b)
    output_root = tmp_path / "ab"
    with pytest.raises(ValueError, match="injected B preflight failure"):
        _prepare_suites(output_root, ["first"])
    assert not output_root.exists()


def test_prepare_many_rolls_back_a_partial_materialization(tmp_path: Path, monkeypatch):
    """Bắt prepare nhiều suite để lại run con dở dang khi suite sau ghi lỗi"""
    calls: list[str] = []

    def _fail_for_b(validated, output_dir, policies):
        calls.append(Path(output_dir).name)
        raise OSError("injected write failure")

    monkeypatch.setattr(observation_cli, "materialize_run", _fail_for_b)
    output_root = tmp_path / "ab"
    with pytest.raises(OSError, match="injected write failure"):
        _prepare_suites(output_root, ["first"])
    assert calls == ["article-42-observation-block-order"]
    assert not (output_root / "a-candidate-position").exists()
    assert not output_root.exists()


def test_prepare_suites_reports_existing_result_without_reading_it(tmp_path: Path):
    """Output root đã tồn tại là boundary chỉ đọc, không suite nào bị materialize."""
    output_root = tmp_path / "ab"
    child = output_root / "candidate-pair-41-3-position"
    child.mkdir(parents=True)
    (child / "manifest.json").write_text("not json", encoding="utf-8")
    outcomes = _prepare_suites(output_root, ["first"])
    assert {outcome.status for outcome in outcomes} == {"existing-result"}
    assert not (output_root / "article-42-observation-block-order").exists()
    assert (child / "manifest.json").read_text(encoding="utf-8") == "not json"
