"""Lưu, reload và kiểm tra artifact fresh-run schema v3 của prompt ablation C.

Bundle C chỉ tham chiếu ba nguồn canonical dùng chung; definition, registry
snapshot, input trace, prompt trace và kết quả nằm hoàn toàn trong artifact của
chính run.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from artifacts import _atomic_write_json, read_snapshot
from contracts import GateCase
from metrics import decision_correctness, selection_correctness
from run_experiments import _utc_now
from seed_cases import read_cases

from diagnostic_subexp.doctoral_defense_prompt_ablation.contracts import (
    PromptAblationInputTrace,
    PromptAblationManifest,
    PromptAblationResult,
    PromptAblationSpec,
    PromptBaselineReference,
    PromptTrace,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.metrics import summarize_prompt_ablation
from diagnostic_subexp.doctoral_defense_prompt_ablation.report import (
    render_prompt_ablation_report,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.schedule import (
    build_prompt_definition,
    build_prompt_input_traces,
    build_prompt_traces,
    plan_prompt_ablation,
    validate_sources,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.variants import prompt_registry
from diagnostic_subexp.shared.contracts import (
    ARTIFACT_ORIGIN,
    ARTIFACT_SCHEMA_VERSION,
    DiagnosticGateRequest,
    SourceReference,
    structured_hash,
)
from diagnostic_subexp.shared.execution import execute_diagnostic_request
from diagnostic_subexp.shared.provenance import execution_provenance
from diagnostic_subexp.shared.source_refs import (
    repository_root,
    sha256_file,
    validate_source_reference,
)

ROOT = repository_root()
C_MODULES = (
    "diagnostic_subexp/shared/contracts.py",
    "diagnostic_subexp/shared/execution.py",
    "diagnostic_subexp/shared/provenance.py",
    "diagnostic_subexp/shared/source_refs.py",
    "diagnostic_subexp/doctoral_defense_prompt_ablation/contracts.py",
    "diagnostic_subexp/doctoral_defense_prompt_ablation/variants.py",
    "diagnostic_subexp/doctoral_defense_prompt_ablation/schedule.py",
    "diagnostic_subexp/doctoral_defense_prompt_ablation/artifacts.py",
    "diagnostic_subexp/doctoral_defense_prompt_ablation/metrics.py",
    "diagnostic_subexp/doctoral_defense_prompt_ablation/report.py",
    "diagnostic_subexp/doctoral_defense_prompt_ablation/cli.py",
)


class PromptAblationArtifactError(ValueError):
    """Báo artifact C đã lưu bị hỏng hoặc không tái lập được."""


def write_manifest(run_dir: Path, manifest: PromptAblationManifest) -> None:
    """Cập nhật manifest C theo cách atomic."""
    _atomic_write_json(Path(run_dir) / "manifest.json", manifest.model_dump(mode="json"))


def read_jsonl(path: Path, model: Any) -> list[Any]:
    """Đọc JSONL nghiêm ngặt vào model Pydantic yêu cầu."""
    rows: list[Any] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise PromptAblationArtifactError(f"{path}: blank line at {line_number}")
        try:
            rows.append(model.model_validate_json(line))
        except ValueError as exc:
            raise PromptAblationArtifactError(f"{path}: invalid row at line {line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: list[Any]) -> None:
    """Ghi model đã validate thành JSONL UTF-8 tất định."""
    Path(path).write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")


def _references(definition: dict[str, Any]) -> list[PromptBaselineReference]:
    """Phân giải mười baseline request identity đã nhúng."""
    return [
        PromptBaselineReference.model_validate(row) for row in definition["baseline_references"]
    ]


def _spec_from_definition(definition: dict[str, Any]) -> PromptAblationSpec:
    """Kiểm tra phần spec của definition C."""
    payload = {key: value for key, value in definition.items() if key in PromptAblationSpec.model_fields}
    return PromptAblationSpec.model_validate(payload)


def _sources(definition: dict[str, Any]) -> dict[str, SourceReference]:
    """Resolve ba nguồn canonical và tính hash từ chính bytes đọc được."""
    sources: dict[str, SourceReference] = {}
    for label in ("case", "snapshot", "inventory"):
        path = definition[f"{label}_path"]
        resolved = ROOT / path
        if not resolved.is_file():
            raise PromptAblationArtifactError(f"canonical {label} source is missing: {path}")
        sources[label] = SourceReference(role=label, path=path, sha256=sha256_file(resolved))
    return sources


def prepare_prompt_ablation(
    output_dir: Path,
    policies: list[str],
    definition: dict[str, Any] | None = None,
) -> tuple[Path, PromptAblationManifest, PromptAblationManifest]:
    """Dựng bằng chứng C fresh schema v3 mà không copy input sinh ra."""
    definition = build_prompt_definition() if definition is None else definition
    spec = _spec_from_definition(definition)
    registry = definition["registry_snapshot"]
    if registry != prompt_registry():
        raise PromptAblationArtifactError(
            "registry snapshot differs from the evaluation-owned registry"
        )
    source_map = _sources(definition)
    case_path = validate_source_reference(source_map["case"])
    snapshot_path = validate_source_reference(source_map["snapshot"])
    cases = read_cases(case_path)
    snapshot = read_snapshot(snapshot_path)
    references = _references(definition)
    validate_sources(spec, cases, snapshot, references, source_map["snapshot"].sha256)
    input_traces = build_prompt_input_traces(references, spec.execution_config)
    traces = build_prompt_traces(
        input_traces,
        references,
        registry,
        spec.execution_config,
        render=True,
    )
    trials = plan_prompt_ablation(spec, cases, input_traces, traces)
    used_ids = {trial.input_trace_id for trial in trials}
    input_traces = [trace for trace in input_traces if trace.input_trace_id in used_ids]
    traces = [trace for trace in traces if trace.input_trace_id in used_ids]
    provenance = execution_provenance(C_MODULES)
    manifest = PromptAblationManifest(
        artifact_origin=ARTIFACT_ORIGIN,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        experiment="fixed-diagnostic",
        diagnostic_kind="defense-fewshot-ablation",
        run_id=uuid.uuid4().hex[:12],
        started_at=_utc_now(),
        status="prepared",
        suite_id=spec.suite_id,
        independent_question_count=len({trial.case_id for trial in trials}),
        policies=policies,
        policy_config={policy: {"status": "not_initialized"} for policy in policies},
        execution_config=spec.execution_config,
        sources=source_map,
        prompt_variants=registry["variants"],
        phases=spec.phases,
        trials=trials,
        input_trace_ids=[trace.input_trace_id for trace in input_traces],
        prompt_trace_ids=[trace.prompt_trace_id for trace in traces],
        exclusions=[
            {"case_id": case.case_id, "reason": "no_candidates"}
            for case in cases
            if not case.candidates
        ],
        expected_trial_count=len(trials),
        expected_policy_result_count=len(trials) * len(policies),
        definition_sha256=structured_hash(definition),
        registry_sha256=structured_hash(registry),
        experiment_definition=definition,
        registry_snapshot=registry,
        **provenance,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        write_jsonl(output_dir / "input_traces.jsonl", input_traces)
        write_jsonl(output_dir / "prompt_traces.jsonl", traces)
        write_manifest(output_dir, manifest)
        load_prompt_ablation(output_dir)
        summary = summarize_prompt_ablation(manifest, cases, input_traces, traces, [])
        summary["status"] = "prepared"
        _atomic_write_json(output_dir / "summary.json", summary)
        (output_dir / "report.md").write_text(
            render_prompt_ablation_report(manifest, traces, summary), encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir, manifest, manifest


def _validate_manifest_sources(manifest: PromptAblationManifest) -> None:
    """Resolve và kiểm tra hash chỉ những nguồn canonical dùng chung."""
    if set(manifest.sources) != {"case", "snapshot", "inventory"}:
        raise PromptAblationArtifactError(
            "fresh manifests must reference exactly case, snapshot and inventory"
        )
    for source in manifest.sources.values():
        try:
            validate_source_reference(source)
        except (OSError, ValueError) as exc:
            raise PromptAblationArtifactError(f"invalid C source reference: {exc}") from exc


def load_prompt_ablation(
    run_dir: Path,
) -> tuple[
    PromptAblationManifest,
    list[GateCase],
    list[PromptAblationInputTrace],
    list[PromptTrace],
    list[PromptAblationResult],
]:
    """Reload fresh run C chỉ từ nguồn canonical và definition đã nhúng."""
    run_dir = Path(run_dir)
    try:
        manifest = PromptAblationManifest.model_validate_json(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise PromptAblationArtifactError(f"invalid C manifest: {exc}") from exc
    _validate_manifest_sources(manifest)
    definition = manifest.experiment_definition
    spec = _spec_from_definition(definition)
    for label, expected_path in (
        ("case", spec.case_path),
        ("snapshot", spec.snapshot_path),
        ("inventory", spec.inventory_path),
    ):
        reference = manifest.sources[label]
        if reference.role != label or reference.path != expected_path:
            raise PromptAblationArtifactError(
                f"source reference {label} disagrees with the embedded definition"
            )
    case_path = validate_source_reference(manifest.sources["case"])
    snapshot_path = validate_source_reference(manifest.sources["snapshot"])
    cases = read_cases(case_path)
    snapshot = read_snapshot(snapshot_path)
    references = _references(definition)
    validate_sources(spec, cases, snapshot, references, manifest.sources["snapshot"].sha256)
    expected_inputs = build_prompt_input_traces(references, spec.execution_config)
    expected_traces = build_prompt_traces(
        expected_inputs,
        references,
        manifest.registry_snapshot,
        spec.execution_config,
    )
    trials = plan_prompt_ablation(spec, cases, expected_inputs, expected_traces)
    used_ids = {trial.input_trace_id for trial in trials}
    expected_inputs = [trace for trace in expected_inputs if trace.input_trace_id in used_ids]
    expected_traces = [trace for trace in expected_traces if trace.input_trace_id in used_ids]
    input_traces = read_jsonl(run_dir / "input_traces.jsonl", PromptAblationInputTrace)
    prompt_traces = read_jsonl(run_dir / "prompt_traces.jsonl", PromptTrace)
    if input_traces != expected_inputs or prompt_traces != expected_traces:
        raise PromptAblationArtifactError("stored C traces differ from embedded evidence")
    if manifest.trials != trials:
        raise PromptAblationArtifactError("stored C schedule differs from embedded definition")
    if manifest.prompt_variants != manifest.registry_snapshot["variants"]:
        raise PromptAblationArtifactError("stored C registry differs from manifest variants")
    results = read_prompt_ablation_results(run_dir, manifest, cases)
    return manifest, cases, input_traces, prompt_traces, results


def derive_result_score(
    case: GateCase,
    presented_candidates: list[int],
    outcome: Any,
) -> dict[str, Any]:
    """Tính lại scoring C từ outcome và nhãn đã duyệt."""
    selected_article: int | None = None
    selected_position: int | None = None
    if outcome.status == "ok" and outcome.decision is not None and not outcome.decision.stop:
        selected_article = outcome.decision.dieu
        if selected_article not in presented_candidates:
            raise PromptAblationArtifactError(
                f"a valid follow selected {selected_article} outside {presented_candidates}"
            )
        selected_position = presented_candidates.index(selected_article) + 1
    return {
        "selected_article": selected_article,
        "selected_position": selected_position,
        "selection_correct": selection_correctness(case, outcome),
        "decision_correct": decision_correctness(case, outcome),
    }


def read_prompt_ablation_results(
    run_dir: Path,
    manifest: PromptAblationManifest,
    cases: list[GateCase],
) -> list[PromptAblationResult]:
    """Kiểm tra từng result C đã lưu với trial và nhãn canonical của nó."""
    trial_by_id = {trial.trial_id: trial for trial in manifest.trials}
    case_by_id = {case.case_id: case for case in cases}
    results: list[PromptAblationResult] = []
    for policy in manifest.policies:
        path = Path(run_dir) / f"results_{policy}.jsonl"
        if not path.exists():
            continue
        seen: set[str] = set()
        for result in read_jsonl(path, PromptAblationResult):
            trial = trial_by_id.get(result.trial_id)
            if trial is None or result.trial_id in seen:
                raise PromptAblationArtifactError("unscheduled or duplicate C result")
            seen.add(result.trial_id)
            if result.policy != policy or result.run_id != manifest.run_id:
                raise PromptAblationArtifactError("C result run or policy mismatch")
            for field in ("input_trace_id", "prompt_trace_id", "request_fingerprint"):
                if getattr(result, field) != getattr(trial, field):
                    raise PromptAblationArtifactError(f"C result {field} mismatch")
            score = derive_result_score(
                case_by_id[trial.case_id], trial.presented_candidates, result.outcome
            )
            for field, value in score.items():
                if getattr(result, field) != value:
                    raise PromptAblationArtifactError(f"C result stale {field}")
            results.append(result)
    return results


def append_prompt_ablation_result(run_dir: Path, result: PromptAblationResult) -> None:
    """Ghi thêm một result C và fsync trước invocation kế tiếp."""
    path = Path(run_dir) / f"results_{result.policy}.jsonl"
    with path.open("a", encoding="utf-8") as target:
        target.write(result.model_dump_json() + "\n")
        target.flush()
        os.fsync(target.fileno())


def execute_prompt_ablation(
    run_dir: Path,
    manifest: PromptAblationManifest,
    cases: list[GateCase],
    input_traces: list[PromptAblationInputTrace],
    prompt_traces: list[PromptTrace],
    clients: dict[str, object],
) -> None:
    """Chạy các request C đã lưu với treatment cô lập theo từng request."""
    inputs = {trace.input_trace_id: trace for trace in input_traces}
    prompts = {trace.prompt_trace_id: trace for trace in prompt_traces}
    case_by_id = {case.case_id: case for case in cases}
    for trial in manifest.trials:
        prompt = prompts[trial.prompt_trace_id]
        source = inputs[trial.input_trace_id]
        if prompt.input_trace_id != source.input_trace_id:
            raise PromptAblationArtifactError("prompt/input binding mismatch")
        request = DiagnosticGateRequest(
            question=source.question,
            observation=source.observation,
            presented_candidates=source.presented_candidates,
            grammar_candidates=source.grammar_candidates,
            grammar_text=source.grammar_text,
            effective_messages=prompt.effective_messages,
        )
        for policy in manifest.policies:
            outcome = execute_diagnostic_request(request, policy=policy, client=clients.get(policy))
            append_prompt_ablation_result(
                run_dir,
                PromptAblationResult(
                    run_id=manifest.run_id,
                    trial_id=trial.trial_id,
                    policy=policy,
                    input_trace_id=source.input_trace_id,
                    prompt_trace_id=prompt.prompt_trace_id,
                    request_fingerprint=prompt.request_fingerprint,
                    outcome=outcome,
                    **derive_result_score(
                        case_by_id[trial.case_id], trial.presented_candidates, outcome
                    ),
                ),
            )


def run_status(
    manifest: PromptAblationManifest,
    results: list[PromptAblationResult],
) -> str:
    """Suy ra trạng thái run C từ các slot đã ghi."""
    scheduled = {
        (policy, trial.trial_id) for policy in manifest.policies for trial in manifest.trials
    }
    completed = {(result.policy, result.trial_id) for result in results}
    if completed != scheduled:
        return "incomplete"
    if any(result.outcome.status == "error" for result in results):
        return "completed_with_errors"
    return "complete"


__all__ = [
    "C_MODULES",
    "PromptAblationArtifactError",
    "append_prompt_ablation_result",
    "derive_result_score",
    "execute_prompt_ablation",
    "load_prompt_ablation",
    "prepare_prompt_ablation",
    "read_jsonl",
    "read_prompt_ablation_results",
    "run_status",
    "write_jsonl",
    "write_manifest",
]
