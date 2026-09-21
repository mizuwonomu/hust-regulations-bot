"""Lưu, reload và kiểm tra artifact fresh-run của intervention A.

Artifact schema v3 chỉ tham chiếu ba nguồn canonical dùng chung; experiment
definition, trace và kết quả nằm hoàn toàn trong bundle kết quả của chính run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from artifacts import _atomic_write_json, read_snapshot
from contracts import GateCase
from metrics import decision_correctness, selection_correctness
from seed_cases import read_cases

from diagnostic_subexp.candidate_pair_41_3_position.contracts import (
    CandidateInputTrace,
    CandidateManifest,
    CandidateSuiteSpec,
    CandidateSummary,
    CandidateTrial,
    DIAGNOSTIC_KIND,
)
from diagnostic_subexp.candidate_pair_41_3_position.schedule import (
    build_candidate_definition,
    verify_candidate_trace,
)
from diagnostic_subexp.shared.contracts import (
    ARTIFACT_ORIGIN,
    ARTIFACT_SCHEMA_VERSION,
    DiagnosticResult,
    sha256_text,
    structured_hash,
)
from diagnostic_subexp.shared.provenance import (
    executed_module_hashes as executed_module_hashes_shared,
    execution_provenance as execution_provenance_shared,
)
from diagnostic_subexp.shared.source_refs import validate_source_reference
from run_experiments import _utc_now

A_MODULES = (
    "diagnostic_subexp/shared/contracts.py",
    "diagnostic_subexp/shared/execution.py",
    "diagnostic_subexp/shared/provenance.py",
    "diagnostic_subexp/shared/source_refs.py",
    "diagnostic_subexp/candidate_pair_41_3_position/contracts.py",
    "diagnostic_subexp/candidate_pair_41_3_position/schedule.py",
    "diagnostic_subexp/candidate_pair_41_3_position/transforms.py",
    "diagnostic_subexp/candidate_pair_41_3_position/artifacts.py",
    "diagnostic_subexp/candidate_pair_41_3_position/metrics.py",
    "diagnostic_subexp/candidate_pair_41_3_position/report.py",
    "diagnostic_subexp/candidate_pair_41_3_position/cli.py",
)


class DiagnosticArtifactError(ValueError):
    """Báo artifact A đã lưu bị hỏng hoặc không tái lập được."""


def executed_module_hashes() -> dict[str, str]:
    """Băm đúng các module package-relative đã thực thi."""
    return executed_module_hashes_shared(list(A_MODULES))


def execution_provenance() -> dict[str, Any]:
    """Ghi revision, dirty state và hash module A đã thực thi."""
    return execution_provenance_shared(A_MODULES)


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except json.JSONDecodeError as exc:
        raise DiagnosticArtifactError(f"{path}: invalid JSON at line {exc.lineno}") from exc
    except OSError as exc:
        raise DiagnosticArtifactError(f"{path}: cannot read artifact: {exc}") from exc


def write_manifest(run_dir: Path, manifest: CandidateManifest) -> None:
    """Cập nhật manifest A theo cách atomic."""
    _atomic_write_json(Path(run_dir) / "manifest.json", manifest.model_dump(mode="json"))


def read_manifest(run_dir: Path) -> CandidateManifest:
    """Đọc và validate manifest A."""
    path = Path(run_dir) / "manifest.json"
    payload = _read_json(path)
    try:
        return CandidateManifest.model_validate(payload)
    except ValueError as exc:
        raise DiagnosticArtifactError(f"{path}: invalid manifest: {exc}") from exc


def write_input_traces(run_dir: Path, traces: list[CandidateInputTrace]) -> None:
    """Ghi toàn bộ input trace theo cách atomic."""
    path = Path(run_dir) / "input_traces.jsonl"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for trace in traces:
            target.write(json.dumps(trace.model_dump(mode="json"), ensure_ascii=False))
            target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def read_input_traces(run_dir: Path) -> list[CandidateInputTrace]:
    """Đọc và validate input trace, từ chối trace trùng hoặc thiếu field."""
    path = Path(run_dir) / "input_traces.jsonl"
    if not path.exists():
        raise DiagnosticArtifactError(f"{path}: required input trace file is missing")
    traces: list[CandidateInputTrace] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise DiagnosticArtifactError(f"{path}: blank line at {line_number}")
            try:
                trace = CandidateInputTrace.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise DiagnosticArtifactError(
                    f"{path}: invalid trace at line {line_number}"
                ) from exc
            if trace.input_trace_id in seen:
                raise DiagnosticArtifactError(
                    f"{path}: duplicate input_trace_id {trace.input_trace_id}"
                )
            seen.add(trace.input_trace_id)
            traces.append(trace)
    return traces


def _result_path(run_dir: Path, policy: str) -> Path:
    return Path(run_dir) / f"results_{policy}.jsonl"


def append_diagnostic_result(run_dir: Path, result: DiagnosticResult) -> None:
    """Thêm một result rồi flush xuống đĩa để run bị ngắt vẫn kiểm tra được."""
    path = _result_path(run_dir, result.policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as target:
        target.write(line)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def create_candidate_run(
    run_dir: Path,
    manifest: CandidateManifest,
    traces: list[CandidateInputTrace],
) -> Path:
    """Tạo run A fresh chỉ với manifest và trace, không có thư mục `sources/`."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise DiagnosticArtifactError(f"{run_dir}: the output directory must be new")
    run_dir.mkdir(parents=True)
    write_input_traces(run_dir, traces)
    write_manifest(run_dir, manifest)
    return run_dir


def _read_result_file(
    path: Path,
    *,
    policy: str,
    manifest: CandidateManifest,
    trial_by_id: dict[str, CandidateTrial],
) -> list[DiagnosticResult]:
    if not path.exists():
        return []
    records: list[DiagnosticResult] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise DiagnosticArtifactError(f"{path}: blank line at {line_number}")
            try:
                record = DiagnosticResult.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise DiagnosticArtifactError(
                    f"{path}: invalid result at line {line_number}"
                ) from exc
            if record.policy != policy:
                raise DiagnosticArtifactError(f"{path}: line {line_number} has policy {record.policy!r}")
            if record.run_id != manifest.run_id:
                raise DiagnosticArtifactError(f"{path}: line {line_number} has a different run_id")
            trial = trial_by_id.get(record.trial_id)
            if trial is None:
                raise DiagnosticArtifactError(
                    f"{path}: line {line_number} references an unscheduled trial"
                )
            if record.input_trace_id != trial.input_trace_id:
                raise DiagnosticArtifactError(
                    f"{path}: line {line_number} references a different input trace"
                )
            if record.request_fingerprint != trial.request_fingerprint:
                raise DiagnosticArtifactError(
                    f"{path}: line {line_number} changed the request fingerprint"
                )
            if record.trial_id in seen:
                raise DiagnosticArtifactError(f"{path}: duplicate result for trial {record.trial_id}")
            seen.add(record.trial_id)
            records.append(record)
    return records


def derive_result_score(
    case: GateCase,
    presented_candidates: list[int],
    outcome: Any,
) -> dict[str, Any]:
    """Tính lại scoring từ outcome và nhãn đã duyệt, không tin cờ đã lưu."""
    selected_article: int | None = None
    selected_position: int | None = None
    if outcome.status == "ok" and outcome.decision is not None and not outcome.decision.stop:
        selected_article = outcome.decision.dieu
        if selected_article not in presented_candidates:
            raise DiagnosticArtifactError(
                f"a valid follow selected {selected_article} outside {presented_candidates}"
            )
        selected_position = presented_candidates.index(selected_article) + 1
    return {
        "selected_article": selected_article,
        "selected_position": selected_position,
        "selection_correct": selection_correctness(case, outcome),
        "decision_correct": decision_correctness(case, outcome),
    }


def read_diagnostic_results(
    run_dir: Path,
    manifest: CandidateManifest,
    cases: list[GateCase],
) -> list[DiagnosticResult]:
    """Đọc result và scoring lại từ nhãn đã đóng băng."""
    case_by_id = {case.case_id: case for case in cases}
    trial_by_id = {trial.trial_id: trial for trial in manifest.trials}
    records: list[DiagnosticResult] = []
    for policy in manifest.policies:
        records.extend(
            _read_result_file(
                _result_path(run_dir, policy),
                policy=policy,
                manifest=manifest,
                trial_by_id=trial_by_id,
            )
        )
    derived: list[DiagnosticResult] = []
    for record in records:
        trial = trial_by_id[record.trial_id]
        case = case_by_id.get(trial.case_id)
        if case is None:
            raise DiagnosticArtifactError(
                f"result for trial {trial.trial_id} references an unknown case"
            )
        score = derive_result_score(case, trial.presented_candidates, record.outcome)
        for field, value in score.items():
            if getattr(record, field) != value:
                raise DiagnosticArtifactError(
                    f"result for trial {trial.trial_id} has a stale {field}: "
                    f"stored {getattr(record, field)!r}, derived {value!r}"
                )
        derived.append(record)
    return derived


def run_status(manifest: CandidateManifest, results: list[DiagnosticResult]) -> str:
    """Suy ra trạng thái run từ các slot đã ghi."""
    scheduled = {
        (policy, trial.trial_id) for policy in manifest.policies for trial in manifest.trials
    }
    completed = {(result.policy, result.trial_id) for result in results}
    if completed != scheduled:
        return "incomplete"
    if any(result.outcome.status == "error" for result in results):
        return "completed_with_errors"
    return "complete"


def verify_trial_trace_binding(
    manifest: CandidateManifest,
    traces: list[CandidateInputTrace],
) -> None:
    """Buộc mỗi trial vào trace hoàn chỉnh mà nó khai báo."""
    trace_ids = [trace.input_trace_id for trace in traces]
    if len(set(trace_ids)) != len(trace_ids):
        raise DiagnosticArtifactError("stored input traces must have unique trace IDs")
    if trace_ids != list(manifest.input_trace_ids):
        raise DiagnosticArtifactError("manifest trace IDs differ from the stored traces")
    trace_by_id = {trace.input_trace_id: trace for trace in traces}
    for trial in manifest.trials:
        trace = trace_by_id.get(trial.input_trace_id)
        if trace is None:
            raise DiagnosticArtifactError(
                f"trial {trial.trial_id} references an unknown input trace"
            )
        for field, bound in (
            ("case_id", trace.case_id),
            ("diagnostic_kind", trace.diagnostic_kind),
            ("arm_id", trace.arm_id),
            ("matched_pair_id", trace.matched_pair_id),
            ("pair_positions", trace.pair_positions),
            ("relative_pair_order", trace.relative_pair_order),
            ("presented_candidates", trace.presented_candidates),
            ("grammar_candidates", trace.grammar_candidates),
            ("observation_hash", trace.transformed_observation_hash),
            ("effective_messages_hash", trace.effective_messages_hash),
            ("grammar_hash", trace.grammar_hash),
            ("request_fingerprint", trace.request_fingerprint),
        ):
            if getattr(trial, field) != bound:
                raise DiagnosticArtifactError(
                    f"trial {trial.trial_id} {field} does not match its bound trace"
                )


def read_candidate_spec_from_definition(payload: dict[str, Any]) -> CandidateSuiteSpec:
    """Kiểm tra definition A đã nhúng trong manifest."""
    return CandidateSuiteSpec.model_validate(payload)


def _validate_embedded_definition(manifest: CandidateManifest) -> CandidateSuiteSpec:
    if set(manifest.sources) != {"case", "snapshot", "inventory"}:
        raise DiagnosticArtifactError(
            "fresh manifests must reference exactly case, snapshot and inventory"
        )
    try:
        spec = read_candidate_spec_from_definition(manifest.experiment_definition)
    except ValueError as exc:
        raise DiagnosticArtifactError(f"invalid embedded experiment definition: {exc}") from exc
    for label, expected_path in (
        ("case", spec.case_path),
        ("snapshot", spec.snapshot_path),
        ("inventory", spec.inventory_path),
    ):
        reference = manifest.sources[label]
        if reference.role != label or reference.path != expected_path:
            raise DiagnosticArtifactError(
                f"source reference {label} disagrees with the embedded definition"
            )
    try:
        for label in ("case", "snapshot", "inventory"):
            validate_source_reference(manifest.sources[label])
    except (OSError, ValueError) as exc:
        raise DiagnosticArtifactError(f"invalid canonical source reference: {exc}") from exc
    return spec


def load_candidate_run(
    run_dir: Path,
) -> tuple[CandidateManifest, list[GateCase], list[CandidateInputTrace], list[DiagnosticResult]]:
    """Reload fresh run A từ nguồn canonical và kiểm tra lại mọi trace bất biến."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    spec = _validate_embedded_definition(manifest)
    case_path = validate_source_reference(manifest.sources["case"])
    snapshot_path = validate_source_reference(manifest.sources["snapshot"])
    cases = read_cases(case_path)
    snapshot = read_snapshot(snapshot_path)

    case_by_id = {case.case_id: case for case in cases}
    source_case = case_by_id.get(manifest.case_id)
    if source_case is None:
        raise DiagnosticArtifactError("canonical case file does not contain the source case")
    if sha256_text(source_case.observation) != source_case.observation_hash:
        raise DiagnosticArtifactError(
            "canonical source case observation hash does not match its bytes"
        )
    if source_case.snapshot_id != snapshot.snapshot_id:
        raise DiagnosticArtifactError("canonical case snapshot id does not match the snapshot")

    traces = read_input_traces(run_dir)
    arm_by_id = {arm.arm_id: arm for arm in spec.arms}
    if set(arm_by_id) != {trace.arm_id for trace in traces}:
        raise DiagnosticArtifactError("stored input traces must cover every declared arm")
    for trace in traces:
        if trace.original_observation_hash != source_case.observation_hash:
            raise DiagnosticArtifactError(
                f"input trace {trace.input_trace_id} original observation hash "
                "differs from the canonical case"
            )
        try:
            verify_candidate_trace(trace, source_case, arm_by_id[trace.arm_id], spec)
        except ValueError as exc:
            raise DiagnosticArtifactError(
                f"input trace {trace.input_trace_id} is not reconstructible: {exc}"
            ) from exc

    verify_trial_trace_binding(manifest, traces)
    results = read_diagnostic_results(run_dir, manifest, cases)
    return manifest, cases, traces, results


def finalize_candidate_run(
    run_dir: Path,
    manifest: CandidateManifest,
    summary: CandidateSummary,
    report_text: str,
) -> None:
    """Lưu summary, report và manifest cuối theo cách atomic."""
    run_dir = Path(run_dir)
    _atomic_write_json(run_dir / "summary.json", summary.model_dump(mode="json"))
    (run_dir / "report.md").write_text(report_text, encoding="utf-8")
    write_manifest(run_dir, manifest)


def build_candidate_manifest(
    *,
    plan: Any,
    policy_config: dict[str, dict[str, Any]],
    run_id: str,
    status: str = "prepared",
) -> CandidateManifest:
    """Dựng manifest fresh-run schema v3 với definition A nhúng sẵn."""
    provenance = execution_provenance()
    base = {
        "case": plan.case_source,
        "snapshot": plan.snapshot_source,
        "inventory": plan.inventory_source,
    }
    definition = build_candidate_definition()
    return CandidateManifest(
        artifact_origin=ARTIFACT_ORIGIN,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        experiment="fixed-diagnostic",
        run_id=run_id,
        started_at=_utc_now(),
        ended_at=None,
        status=status,
        suite_id=plan.suite_id,
        diagnostic_kind=DIAGNOSTIC_KIND,
        case_id=plan.source_case_identity.case_id,
        independent_question_count=1,
        sources=base,
        execution_revision=provenance["execution_revision"],
        execution_dirty=provenance["execution_dirty"],
        executed_module_hashes=provenance["executed_module_hashes"],
        policies=list(plan.policies),
        execution_config=plan.execution_config,
        policy_config=policy_config,
        trials=list(plan.trials),
        input_trace_ids=[trace.input_trace_id for trace in plan.traces],
        exclusions=list(plan.exclusions),
        expected_trial_count=plan.expected_trial_count,
        expected_policy_result_count=plan.expected_policy_result_count,
        definition_sha256=structured_hash(definition),
        experiment_definition=definition,
    )


__all__ = [
    "A_MODULES",
    "DiagnosticArtifactError",
    "append_diagnostic_result",
    "build_candidate_manifest",
    "create_candidate_run",
    "derive_result_score",
    "executed_module_hashes",
    "execution_provenance",
    "finalize_candidate_run",
    "load_candidate_run",
    "read_candidate_spec_from_definition",
    "read_diagnostic_results",
    "read_input_traces",
    "read_manifest",
    "run_status",
    "verify_trial_trace_binding",
    "write_input_traces",
    "write_manifest",
]
