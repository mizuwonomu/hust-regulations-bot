"""Kiểm tra runner dual-suite A/B của primary và tính độc lập của hai suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

AGENT_EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_EXP_ROOT.parents[1]
sys.path.insert(0, str(AGENT_EXP_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from diagnostic_subexp.article_42_observation_block_order import cli as observation_cli
from diagnostic_subexp.candidate_pair_41_3_position import cli as run_diagnostics
from run_fixed_diagnostics_ab import HARNESSES
import run_fixed_diagnostics_ab

WRAPPER_PATH = AGENT_EXP_ROOT / "scripts" / "run_fixed_diagnostics_ab.py"
A_CHILD = "candidate-pair-41-3-position"
B_CHILD = "article-42-observation-block-order"


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch):
    """Chạy mọi test từ repository root như cách vận hành thật."""
    monkeypatch.chdir(REPO_ROOT)
    yield


def _cli(*args: str) -> subprocess.CompletedProcess:
    """Chạy wrapper trực tiếp từ repository root."""
    return subprocess.run(
        [sys.executable, str(WRAPPER_PATH), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _read_json(path: Path) -> dict:
    """Đọc artifact JSON đã lưu."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_direct_wrapper_help_bootstraps_repository_imports():
    """Bắt wrapper direct cần pytest path injection mới chạy được."""
    completed = _cli("--help")
    assert completed.returncode == 0, completed.stderr
    assert "prepare" in completed.stdout
    assert "summarize" in completed.stdout


def test_harnesses_are_only_the_two_semantic_packages():
    """Hai harness con phải tách biệt, không dùng chung module A."""
    assert set(HARNESSES) == {A_CHILD, B_CHILD}
    assert HARNESSES[A_CHILD] is run_diagnostics
    assert HARNESSES[B_CHILD] is observation_cli


def test_prepare_creates_two_deterministic_child_directories(tmp_path: Path):
    """Bắt prepare gộp hai suite hoặc đặt tên thư mục con không tất định."""
    output_root = tmp_path / "ab-run"
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(output_root), "--policies", "first"]
    ) == 0

    assert {path.name for path in output_root.iterdir()} == {A_CHILD, B_CHILD}
    for child, suite_id in (
        (A_CHILD, "fixed-diagnostic-a-v1"),
        (B_CHILD, "fixed-diagnostic-b-v1"),
    ):
        manifest = _read_json(output_root / child / "manifest.json")
        assert manifest["suite_id"] == suite_id
        assert manifest["status"] == "prepared"
        assert manifest["policies"] == ["first"]
        assert (output_root / child / "input_traces.jsonl").exists()
        assert not (output_root / child / "sources").exists()
        assert not (output_root / "manifest.json").exists()


def test_prepare_reports_existing_result_on_an_existing_output_root(tmp_path: Path):
    """Output root đã tồn tại thì prepare báo existing-result thành công, không ghi."""
    output_root = tmp_path / "ab-run"
    output_root.mkdir()
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(output_root), "--policies", "first"]
    ) == 0
    assert not (output_root / A_CHILD).exists()
    assert not (output_root / B_CHILD).exists()


def test_first_policy_lifecycle_completes_both_suites(tmp_path: Path):
    """Bắt một phase làm mất slot, mất summary hoặc gộp metric A/B."""
    output_root = tmp_path / "ab-run"
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(output_root), "--policies", "first"]
    ) == 0
    assert run_fixed_diagnostics_ab.main(["run", "--output-root", str(output_root)]) == 0

    expected = {A_CHILD: 30, B_CHILD: 12}
    for child, count in expected.items():
        manifest = _read_json(output_root / child / "manifest.json")
        assert manifest["status"] == "complete"
        assert manifest["ended_at"] is not None
        assert manifest["expected_trial_count"] == count
        results = [
            json.loads(line)
            for line in (output_root / child / "results_first.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(results) == count
        assert {result["policy"] for result in results} == {"first"}
        assert (output_root / child / "summary.json").exists()
        assert (output_root / child / "report.md").exists()
        assert not (output_root / child / "results_llm.jsonl").exists()

    assert run_fixed_diagnostics_ab.main(["summarize", "--output-root", str(output_root)]) == 0
    for child, count in expected.items():
        summary = _read_json(output_root / child / "summary.json")
        assert summary["scheduled"] == count
        assert summary["valid"] == count
        assert summary["missing"] == 0


def test_summarize_is_offline_for_a_prepared_root(tmp_path: Path):
    """Bắt summarize gọi model hoặc giả STOP cho slot còn thiếu."""
    output_root = tmp_path / "ab-run"
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(output_root), "--policies", "first"]
    ) == 0
    assert run_fixed_diagnostics_ab.main(["summarize", "--output-root", str(output_root)]) == 0

    for child, count in ((A_CHILD, 30), (B_CHILD, 12)):
        manifest = _read_json(output_root / child / "manifest.json")
        assert manifest["status"] == "prepared"
        summary = _read_json(output_root / child / "summary.json")
        assert summary["scheduled"] == count
        assert summary["missing"] == count
        assert summary["valid"] == 0


def test_prepare_never_constructs_an_llm_client(monkeypatch, tmp_path: Path):
    """Bắt prepare dựng model client cho policy llm."""
    import policies.llm

    def _refuse(*_args, **_kwargs):
        raise AssertionError("prepare must not construct an LLM client")

    monkeypatch.setattr(policies.llm, "create_client", _refuse)
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(tmp_path / "ab-run"), "--policies", "first", "llm"]
    ) == 0


def test_run_reports_missing_children_without_merging(tmp_path: Path):
    """Bắt run im lặng thành công khi một suite con chưa được prepare."""
    output_root = tmp_path / "ab-run"
    output_root.mkdir()
    assert run_fixed_diagnostics_ab.main(["run", "--output-root", str(output_root)]) == 1
    assert run_fixed_diagnostics_ab.main(["summarize", "--output-root", str(output_root)]) == 1


def test_aggregate_run_fails_when_one_suite_fails(tmp_path: Path, monkeypatch):
    """Bắt lỗi của A làm mất trạng thái của B hoặc bị báo là thành công."""
    output_root = tmp_path / "ab-run"
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(output_root), "--policies", "first"]
    ) == 0

    def _fail_for_a(run_dir: Path):
        if Path(run_dir).name == A_CHILD:
            raise ValueError("injected A failure")
        return observation_cli.execute_run(run_dir)

    monkeypatch.setattr(run_diagnostics, "execute_run", _fail_for_a)
    assert run_fixed_diagnostics_ab.main(["run", "--output-root", str(output_root)]) == 1

    assert _read_json(output_root / A_CHILD / "manifest.json")["status"] in {
        "prepared",
        "incomplete",
    }
    assert _read_json(output_root / B_CHILD / "manifest.json")["status"] == "complete"
    assert len(
        (output_root / B_CHILD / "results_first.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 12
    assert not (output_root / A_CHILD / "results_first.jsonl").exists()


def test_keyboard_interrupt_stops_orchestration_with_130(tmp_path: Path, monkeypatch):
    """Bắt keyboard interrupt bị đổi thành exit code khác 130."""
    output_root = tmp_path / "ab-run"
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(output_root), "--policies", "first"]
    ) == 0

    def _interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_diagnostics, "execute_run", _interrupt)
    assert run_fixed_diagnostics_ab.main(["run", "--output-root", str(output_root)]) == 130
    # Suite A phải được finalize trước khi re-raise, không mắc kẹt ở running
    assert _read_json(output_root / A_CHILD / "manifest.json")["status"] != "running"
    assert _read_json(output_root / B_CHILD / "manifest.json")["status"] == "prepared"


def test_dual_suite_children_keep_independent_trace_identity(tmp_path: Path):
    """Bắt wrapper dùng chung trace id hoặc trộn analysis group giữa A và B."""
    output_root = tmp_path / "ab-run"
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(output_root), "--policies", "first"]
    ) == 0

    a_manifest = _read_json(output_root / A_CHILD / "manifest.json")
    b_manifest = _read_json(output_root / B_CHILD / "manifest.json")
    assert set(a_manifest["input_trace_ids"]).isdisjoint(b_manifest["input_trace_ids"])

    a_control = next(
        trace
        for trace in map(
            json.loads,
            (output_root / A_CHILD / "input_traces.jsonl").read_text(encoding="utf-8").splitlines(),
        )
        if trace["arm_id"] == "a0-original-control"
    )
    b_control = next(
        trace
        for trace in map(
            json.loads,
            (output_root / B_CHILD / "input_traces.jsonl").read_text(encoding="utf-8").splitlines(),
        )
        if trace["arm_id"] == "b0-original-control"
    )
    assert a_control["request_fingerprint"] == b_control["request_fingerprint"]
    assert a_control["input_trace_id"] != b_control["input_trace_id"]


def test_single_suite_runner_still_works_from_repository_root(tmp_path: Path):
    """Bắt tích hợp primary làm hỏng entrypoint single-suite đã nhận."""
    run_dir = tmp_path / "a-run"
    assert run_diagnostics.main(
        ["prepare", "--output-dir", str(run_dir), "--policies", "first"]
    ) == 0
    assert run_diagnostics.main(["run", "--run-dir", str(run_dir)]) == 0
    assert _read_json(run_dir / "manifest.json")["status"] == "complete"


def test_duplicate_policy_selection_is_rejected(tmp_path: Path):
    """Bắt wrapper chấp nhận policy lặp và nhân đôi slot."""
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(tmp_path / "ab-run"), "--policies", "first", "first"]
    ) == 2
    assert not (tmp_path / "ab-run").exists()


def test_failed_schedule_is_finalized_instead_of_stuck_running(tmp_path: Path, monkeypatch):
    """Bắt wrapper để manifest mắc kẹt ở running khi schedule lỗi giữa chừng."""
    output_root = tmp_path / "ab-run"
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(output_root), "--policies", "first"]
    ) == 0

    for module in (run_diagnostics, observation_cli):
        original = module.execute_schedule
        calls: list[str] = []

        def _fail_after_full_pass(
            run_dir, manifest, cases, traces, clients, _original=original, _calls=calls
        ):
            _original(run_dir, manifest, cases, traces, clients)
            _calls.append(Path(run_dir).name)
            raise RuntimeError("injected mid-schedule failure")

        monkeypatch.setattr(module, "execute_schedule", _fail_after_full_pass)

    outcomes = run_fixed_diagnostics_ab._run_suites(output_root)
    assert {outcome.name: outcome.status for outcome in outcomes} == {
        A_CHILD: "failed",
        B_CHILD: "failed",
    }

    for child in (A_CHILD, B_CHILD):
        manifest = _read_json(output_root / child / "manifest.json")
        assert manifest["status"] in {"complete", "completed_with_errors", "incomplete"}
        assert manifest["status"] != "running"
        assert manifest["ended_at"] is not None
        assert (output_root / child / "summary.json").exists()


def test_interrupted_schedule_is_finalized_before_re_raising(tmp_path: Path, monkeypatch):
    """Bắt keyboard interrupt giữa schedule để lại manifest running dở dang."""
    output_root = tmp_path / "ab-run"
    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(output_root), "--policies", "first"]
    ) == 0

    def _interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_diagnostics, "execute_schedule", _interrupt)
    assert run_fixed_diagnostics_ab.main(["run", "--output-root", str(output_root)]) == 130

    interrupted = _read_json(output_root / A_CHILD / "manifest.json")
    assert interrupted["status"] == "incomplete"
    assert interrupted["ended_at"] is not None
    assert (output_root / A_CHILD / "report.md").exists()
    # B chưa từng được chạy nên phải giữ nguyên trạng thái prepared
    assert _read_json(output_root / B_CHILD / "manifest.json")["status"] == "prepared"


def test_prepare_is_atomic_when_one_suite_fails_preflight(tmp_path: Path, monkeypatch):
    """Bắt prepare để lại suite A đã materialize khi suite B preflight thất bại."""

    def _fail_for_b(_spec_path, _policies):
        raise ValueError("injected B preflight failure")

    monkeypatch.setattr(observation_cli, "validate_run", _fail_for_b)
    output_root = tmp_path / "ab-run"

    assert run_fixed_diagnostics_ab.main(
        ["prepare", "--output-root", str(output_root), "--policies", "first"]
    ) == 2
    assert not output_root.exists()


def test_manifest_records_the_dirty_execution_state(tmp_path: Path):
    """Bắt manifest gắn revision sạch trong khi source đang dirty ngoài HEAD."""
    from diagnostic_subexp.candidate_pair_41_3_position.artifacts import execution_provenance

    output_root = tmp_path / "ab-run"
    manifest = run_diagnostics.prepare_run(None, output_root / A_CHILD, ["first"]).manifest
    assert manifest.execution_revision is not None
    assert manifest.execution_dirty is execution_provenance()["execution_dirty"]
    assert manifest.model_dump(mode="json")["execution_dirty"] is manifest.execution_dirty
