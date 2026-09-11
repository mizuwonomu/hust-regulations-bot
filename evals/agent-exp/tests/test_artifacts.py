"""Kiểm tra persistence offline và reload của experiment artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from artifacts import (
    ArtifactCorruptionError,
    append_debug_record,
    append_result,
    create_run,
    finalize_run,
    load_run,
    read_snapshot,
    sha256_file,
    write_snapshot,
)
from contracts import GateCase, PermutationConfig, RunManifest, SourceFile, Trial
from metrics import score_result, summarize
from permutation import build_permutation_plan
from seed_cases import prepare_cases, write_cases
from src.rag.agent.schema import Decision
from contracts import DecisionOutcome


def _inputs(tmp_path: Path):
    from contracts import SeedRow, SeedSnapshot

    snapshot = SeedSnapshot(
        schema_version=1,
        snapshot_id="snapshot",
        dataset_id="dataset",
        baseline_source=SourceFile(path="baseline.json", sha256="baseline-hash"),
        dataset_source=SourceFile(path="dataset.json", sha256="dataset-hash"),
        whitelist_source=SourceFile(path="whitelist.json", sha256="whitelist-hash"),
        retrieval_config={},
        unavailable_metadata={},
        internal_dieu={10, 20},
        rows=[
            SeedRow(
                id=1,
                question="question",
                contexts=["Điều 10. A\nTheo Điều 20"],
                seed_dieu={10},
            )
        ],
    )
    snapshot_path = tmp_path / "snapshot.json"
    write_snapshot(snapshot, snapshot_path)
    snapshot_hash = sha256_file(snapshot_path)
    draft = prepare_cases(snapshot, snapshot_hash=snapshot_hash)[0]
    case = GateCase.model_validate(
        {
            **draft.model_dump(mode="python"),
            "expected_action": "follow",
            "acceptable_dieu": [20],
            "label_reason": "Điều 20 cần cho câu hỏi",
            "label_status": "approved",
        }
    )
    cases_path = tmp_path / "cases.jsonl"
    write_cases([case], cases_path)
    trial = Trial(
        trial_id="trial",
        case_id=case.case_id,
        repeat_id=0,
        condition="original",
        permutation_id="original",
        seed_order=[],
        candidate_order=[20],
        observation_hash=case.observation_hash,
    )
    manifest = RunManifest(
        schema_version=1,
        run_id="run",
        experiment="initial-selection",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at=None,
        status="running",
        policies=["first"],
        trials=[trial],
        cases_source=SourceFile(path=str(cases_path), sha256=sha256_file(cases_path)),
        snapshot_source=SourceFile(path=str(snapshot_path), sha256=sha256_file(snapshot_path)),
        exclusions=[],
        provenance={},
        policy_config={"first": {}},
    )
    return snapshot_path, cases_path, case, trial, manifest


def test_run_artifacts_reload_without_debug_files(tmp_path):
    snapshot_path, cases_path, case, trial, manifest = _inputs(tmp_path)
    run_dir = create_run(tmp_path / "results", manifest)
    result = score_result(
        "run",
        "first",
        trial,
        case,
        DecisionOutcome(
            status="ok",
            decision=Decision(stop=False, dieu=20),
            error=None,
            latency_ms=1.0,
            usage=None,
        ),
    )
    append_result(run_dir, result)
    loaded_manifest, loaded_cases, loaded_results = load_run(run_dir)
    assert loaded_manifest.run_id == "run"
    assert loaded_cases[0].case_id == case.case_id
    assert loaded_results[0].trial.trial_id == "trial"

    summary = summarize(loaded_manifest, loaded_cases, loaded_results)
    finalize_run(run_dir, loaded_manifest.model_copy(update={"status": "complete"}), summary)
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert not (run_dir / "debug").exists()


def test_run_directory_never_overwrites(tmp_path, monkeypatch):
    # Khóa path để buộc collision trong cùng test và bảo vệ manifest cũ
    import artifacts

    _, _, _, _, manifest = _inputs(tmp_path)
    output_root = tmp_path / "results"
    monkeypatch.setattr(artifacts, "_run_directory_name", lambda current: "fixed-run")
    first = create_run(output_root, manifest)
    original = "manifest goc"
    (first / "manifest.json").write_text(original, encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_run(output_root, manifest.model_copy(update={"run_id": "run-2"}))
    assert (first / "manifest.json").read_text(encoding="utf-8") == original


def _two_candidate_inputs(tmp_path: Path):
    from contracts import SeedRow, SeedSnapshot

    snapshot = SeedSnapshot(
        schema_version=1,
        snapshot_id="snapshot",
        dataset_id="dataset",
        baseline_source=SourceFile(path="baseline.json", sha256="baseline-hash"),
        dataset_source=SourceFile(path="dataset.json", sha256="dataset-hash"),
        whitelist_source=SourceFile(path="whitelist.json", sha256="whitelist-hash"),
        retrieval_config={},
        unavailable_metadata={},
        internal_dieu={10, 20, 30},
        rows=[
            SeedRow(
                id=1,
                question="question",
                contexts=["Điều 10. A\nTheo Điều 20 và Điều 30"],
                seed_dieu={10},
            )
        ],
    )
    snapshot_path = tmp_path / "snapshot.json"
    write_snapshot(snapshot, snapshot_path)
    snapshot_hash = sha256_file(snapshot_path)
    draft = prepare_cases(snapshot, snapshot_hash=snapshot_hash)[0]
    case = GateCase.model_validate(
        {
            **draft.model_dump(mode="python"),
            "expected_action": "follow",
            "acceptable_dieu": [20],
            "label_reason": "Điều 20 cần cho câu hỏi",
            "label_status": "approved",
        }
    )
    cases_path = tmp_path / "cases.jsonl"
    write_cases([case], cases_path)
    trial = Trial(
        trial_id="trial",
        case_id=case.case_id,
        repeat_id=0,
        condition="original",
        permutation_id="original",
        seed_order=[],
        candidate_order=[20, 30],
        observation_hash=case.observation_hash,
    )
    manifest = RunManifest(
        schema_version=1,
        run_id="run",
        experiment="initial-selection",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at=None,
        status="running",
        policies=["first"],
        trials=[trial],
        cases_source=SourceFile(path=str(cases_path), sha256=sha256_file(cases_path)),
        snapshot_source=SourceFile(path=str(snapshot_path), sha256=sha256_file(snapshot_path)),
        exclusions=[],
        provenance={},
        policy_config={"first": {}},
    )
    return case, trial, manifest


def _rewrite_single_result(run_dir: Path, mutate) -> None:
    path = run_dir / "results_first.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_stop_on_follow_round_trips_through_reload_and_summary(tmp_path):
    case, trial, manifest = _two_candidate_inputs(tmp_path)
    run_dir = create_run(tmp_path / "results", manifest)
    result = score_result(
        "run",
        "first",
        trial,
        case,
        DecisionOutcome(
            status="ok",
            decision=Decision(stop=True, dieu=None),
            error=None,
            latency_ms=1.0,
            usage=None,
        ),
    )
    assert result.selection_correct is False
    append_result(run_dir, result)

    loaded_manifest, loaded_cases, loaded_results = load_run(run_dir)
    assert len(loaded_results) == 1
    assert loaded_results[0].selection_correct is False
    assert loaded_results[0].decision_correct is False
    summary = summarize(loaded_manifest, loaded_cases, loaded_results).by_policy["first"]
    assert summary.valid == 1
    assert summary.errors == 0
    assert summary.decision_accuracy.value == 0.0


def test_mutated_correctness_flags_are_rejected_on_reload(tmp_path):
    case, trial, manifest = _two_candidate_inputs(tmp_path)
    run_dir = create_run(tmp_path / "results", manifest)
    result = score_result(
        "run",
        "first",
        trial,
        case,
        DecisionOutcome(
            status="ok",
            decision=Decision(stop=False, dieu=30),
            error=None,
            latency_ms=1.0,
            usage=None,
        ),
    )
    assert result.selection_correct is False
    append_result(run_dir, result)

    def _cheat(payload):
        payload["selection_correct"] = True
        payload["decision_correct"] = True

    # Cố tình sửa correctness nhưng giữ nguyên quyết định để kiểm tra tính nhất quán của record
    _rewrite_single_result(run_dir, _cheat)

    with pytest.raises(ArtifactCorruptionError, match="invalid result"):
        load_run(run_dir)


def test_mutated_label_is_rejected_at_summary(tmp_path):
    case, trial, manifest = _two_candidate_inputs(tmp_path)
    run_dir = create_run(tmp_path / "results", manifest)
    result = score_result(
        "run",
        "first",
        trial,
        case,
        DecisionOutcome(
            status="ok",
            decision=Decision(stop=False, dieu=30),
            error=None,
            latency_ms=1.0,
            usage=None,
        ),
    )
    append_result(run_dir, result)

    def _relabel(payload):
        payload["expected_action"] = "stop"
        payload["acceptable_dieu"] = []
        payload["selection_correct"] = None
        payload["decision_correct"] = False

    # Record tự nhất quán vẫn phải khớp nhãn trong case nguồn đã duyệt
    _rewrite_single_result(run_dir, _relabel)

    loaded_manifest, loaded_cases, loaded_results = load_run(run_dir)
    assert len(loaded_results) == 1
    with pytest.raises(ValueError, match="label differs"):
        summarize(loaded_manifest, loaded_cases, loaded_results)


def test_truncated_result_is_corruption_not_model_error(tmp_path):
    _, _, _, _, manifest = _inputs(tmp_path)
    run_dir = create_run(tmp_path / "results", manifest)
    result_path = run_dir / "results_first.jsonl"
    result_path.write_text('{"run_id":', encoding="utf-8")

    with pytest.raises(ArtifactCorruptionError, match="invalid result"):
        load_run(run_dir)


def test_reload_rejects_duplicate_result_and_outside_schedule(tmp_path):
    _, _, case, trial, manifest = _inputs(tmp_path)
    run_dir = create_run(tmp_path / "results", manifest)
    result = score_result(
        "run",
        "first",
        trial,
        case,
        DecisionOutcome(status="ok", decision=Decision(stop=False, dieu=20), error=None, latency_ms=1.0, usage=None),
    )
    append_result(run_dir, result)
    append_result(run_dir, result)
    # Khóa kết quả là policy và trial_id, mọi record phải thuộc schedule đã lưu
    with pytest.raises(ArtifactCorruptionError, match="duplicate result"):
        load_run(run_dir)

    run_dir = create_run(tmp_path / "results-2", manifest)
    outsider_trial = trial.model_copy(update={"trial_id": "outside"})
    outsider = score_result("run", "first", outsider_trial, case, result.outcome)
    append_result(run_dir, outsider)
    with pytest.raises(ArtifactCorruptionError, match="unknown trial"):
        load_run(run_dir)


def test_reload_rejects_changed_case_or_missing_input(tmp_path):
    snapshot_path, cases_path, case, trial, manifest = _inputs(tmp_path)
    run_dir = create_run(tmp_path / "results", manifest)
    result = score_result("run", "first", trial, case, DecisionOutcome(status="ok", decision=Decision(stop=False, dieu=20), error=None, latency_ms=1.0, usage=None))
    append_result(run_dir, result)
    # Thay đổi input sau run phải bị phát hiện bằng hash thay vì chấm lại trên nhãn mới
    cases_path.write_text(cases_path.read_text(encoding="utf-8").replace("Điều 20 cần cho câu hỏi", "changed"), encoding="utf-8")
    with pytest.raises(ArtifactCorruptionError, match="source hash mismatch"):
        load_run(run_dir)

    cases_path.write_text(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n", encoding="utf-8")
    snapshot_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_run(run_dir)


def test_snapshot_membership_serialization_is_sorted(tmp_path):
    snapshot_path, _, _, _, _ = _inputs(tmp_path)
    snapshot = read_snapshot(snapshot_path)
    row = snapshot.rows[0].model_copy(update={"seed_dieu": {20, 10}})
    snapshot = snapshot.model_copy(update={"internal_dieu": {20, 10}, "rows": [row]})
    write_snapshot(snapshot, snapshot_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["internal_dieu"] == sorted(payload["internal_dieu"])
    assert payload["rows"][0]["seed_dieu"] == sorted(payload["rows"][0]["seed_dieu"])


def test_read_cases_rejects_duplicate_case_id(tmp_path):
    snapshot_path, _, case, _, _ = _inputs(tmp_path)
    cases_path = tmp_path / "duplicate-cases.jsonl"
    write_cases([case, case], cases_path)
    from seed_cases import read_cases

    cases = read_cases(cases_path)
    with pytest.raises(ValueError, match="duplicate"):
        from seed_cases import validate_replay

        validate_replay(
            read_snapshot(snapshot_path),
            cases,
            snapshot_hash=sha256_file(snapshot_path),
        )


def test_separate_policy_files_share_trial_identity(tmp_path):
    snapshot_path, cases_path, case, trial, manifest = _inputs(tmp_path)
    manifest = manifest.model_copy(update={"policies": ["first", "llm"], "policy_config": {"first": {}, "llm": {}}})
    run_dir = create_run(tmp_path / "results", manifest)
    first = score_result("run", "first", trial, case, DecisionOutcome(status="ok", decision=Decision(stop=False, dieu=20), error=None, latency_ms=1.0, usage=None))
    llm = score_result("run", "llm", trial, case, DecisionOutcome(status="ok", decision=Decision(stop=True, dieu=None), error=None, latency_ms=2.0, usage=None))
    append_result(run_dir, first)
    append_result(run_dir, llm)
    loaded_manifest, _, results = load_run(run_dir)
    assert loaded_manifest.policies == ["first", "llm"]
    assert {record.policy for record in results} == {"first", "llm"}
    assert {record.trial.trial_id for record in results} == {"trial"}


def test_summary_recomputes_after_fixture_debug_deletion(tmp_path):
    _, _, case, trial, manifest = _inputs(tmp_path)
    run_dir = create_run(tmp_path / "results", manifest)
    result = score_result("run", "first", trial, case, DecisionOutcome(status="ok", decision=Decision(stop=False, dieu=20), error=None, latency_ms=1.0, usage=None))
    append_result(run_dir, result)
    append_debug_record(run_dir, "first", {"question": case.question, "raw": "fixture"})
    loaded_manifest, loaded_cases, loaded_results = load_run(run_dir)
    before = summarize(loaded_manifest, loaded_cases, loaded_results)
    # Xóa debug của fixture rồi tính lại để chứng minh compact artifacts đủ tái lập summary
    shutil.rmtree(run_dir / "debug")
    after_manifest, after_cases, after_results = load_run(run_dir)
    assert summarize(after_manifest, after_cases, after_results) == before


def test_repository_ignore_rules_keep_debug_ignored_and_results_tracked():
    # Kiểm tra ignore rules thật và bảo đảm results JSONL không bị bỏ qua
    debug = subprocess.run(
        ["git", "check-ignore", "--no-index", "evals/agent-exp/results/x/debug/decisions_first.jsonl"],
        capture_output=True,
        text=True,
        check=False,
    )
    log = subprocess.run(
        ["git", "check-ignore", "--no-index", "evals/agent-exp/results/x/run_first.log"],
        capture_output=True,
        text=True,
        check=False,
    )
    compact = subprocess.run(
        ["git", "check-ignore", "--no-index", "evals/agent-exp/results/x/results_first.jsonl"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert debug.returncode == 0
    assert log.returncode == 0
    assert compact.returncode == 1


def _permutation_run_inputs(tmp_path: Path):
    case, _, initial_manifest = _two_candidate_inputs(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"
    cases_path = tmp_path / "cases.jsonl"
    snapshot = read_snapshot(snapshot_path)
    config = PermutationConfig(condition="candidate-order", schedule="rotate", repeats=1)
    plan = build_permutation_plan([case], snapshot, config, policy_count=1)
    manifest = initial_manifest.model_copy(
        update={
            "schema_version": 2,
            "experiment": "permutation",
            "trials": plan.trials,
            "permutation": config,
            "expected_trial_count": plan.expected_trial_count,
            "expected_policy_result_count": plan.expected_policy_result_count,
        }
    )
    return snapshot_path, cases_path, case, manifest


def test_permutation_reload_rejects_tampered_candidate_schedule(tmp_path):
    _, _, case, manifest = _permutation_run_inputs(tmp_path)
    run_dir = create_run(tmp_path / "results", manifest)
    for trial in manifest.trials:
        append_result(
            run_dir,
            score_result(
                "run",
                "first",
                trial,
                case,
                DecisionOutcome(
                    status="ok",
                    decision=Decision(stop=False, dieu=trial.candidate_order[0]),
                    error=None,
                    latency_ms=1.0,
                    usage=None,
                ),
            ),
        )
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["trials"][0]["candidate_order"] = [30, 20]
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ArtifactCorruptionError, match="schedule"):
        load_run(run_dir)
