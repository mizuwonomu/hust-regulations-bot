"""Tạo run STOP mới, ghi result tăng dần, reload nghiêm ngặt và provenance thực thi

Run chỉ tham chiếu nguồn canonical bên ngoài; không có bản sao case/snapshot hay
trajectory nào nằm trong thư mục kết quả
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

from artifacts import _atomic_write_json, read_snapshot
from contracts import SeedSnapshot

from diagnostic_subexp.shared.provenance import sha256_file
from diagnostic_subexp.shared.source_refs import repository_root

from stop_policy_eval.cases import read_stop_cases, resolve_source_file
from stop_policy_eval.contracts import (
    STOP_POLICY_SCHEMA_VERSION,
    StopInputTrace,
    StopPlan,
    StopPolicyCase,
    StopResult,
    StopRunManifest,
    StopSummary,
)
from stop_policy_eval.execution import (
    PREPARED_CONFIG_REASON,
    effective_execution_config,
    execute_stop_request,
    render_baseline_trace,
    request_fingerprint,
)
from stop_policy_eval.metrics import derive_result_score, summarize_stop_run
from stop_policy_eval.report import render_source_capture_report, render_stop_report
from stop_policy_eval.schedule import build_stop_plan
from stop_policy_eval.support import sanitize_error, utc_now

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
REPO_ROOT = repository_root()
STOP_PACKAGE_REQUIRED_FILES: tuple[str, ...] = (
    "stop_policy_eval/__init__.py",
    "stop_policy_eval/contracts.py",
    "stop_policy_eval/cases.py",
    "stop_policy_eval/capture.py",
    "stop_policy_eval/schedule.py",
    "stop_policy_eval/execution.py",
    "stop_policy_eval/artifacts.py",
    "stop_policy_eval/metrics.py",
    "stop_policy_eval/report.py",
    "stop_policy_eval/cli.py",
    "stop_policy_eval/support.py",
    "stop_policy_eval/state.py",
    "stop_policy_eval/later_hop.py",
)
DEPENDENCY_MODULES: tuple[str, ...] = (
    "artifacts",
    "contracts",
    "seed_cases",
    "seed_capture",
    "run_experiments",
    "diagnostic_subexp.shared.contracts",
    "diagnostic_subexp.shared.execution",
    "diagnostic_subexp.shared.provenance",
    "diagnostic_subexp.shared.source_refs",
    "policies.first",
    "policies.llm",
    "evals.common.single_pass_retrieval",
    "src.rag.agent.prompt",
    "src.rag.agent.gate",
    "src.rag.agent.schema",
    "src.rag.agent.llm_client",
    "src.rag.config",
    "src.rag.agent.loop",
    "src.rag.agent.tools",
    "src.ingestion.reference_parser",
)
EXECUTED_MODULES: tuple[str, ...] = tuple(
    path.relative_to(SCRIPTS_ROOT).as_posix() for path in sorted(SCRIPT_DIR.glob("*.py"))
)
MANIFEST_FILE = "manifest.json"
TRACES_FILE = "input_traces.jsonl"
SUMMARY_FILE = "summary.json"
REPORT_FILE = "report.md"


class StopArtifactError(ValueError):
    """Báo artifact STOP đã lưu bị hỏng hoặc không tái lập được"""


def _source_module_path(module_name: str) -> Path:
    """Resolve source của module local mà không import module chưa được gọi"""
    loaded = sys.modules.get(module_name)
    loaded_path = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_path is not None:
        path = Path(loaded_path).resolve()
    else:
        relative = Path(*module_name.split(".")).with_suffix(".py")
        candidates = [
            (root / relative).resolve()
            for root in (SCRIPTS_ROOT, REPO_ROOT)
            if (root / relative).is_file()
        ]
        if len(candidates) != 1:
            raise StopArtifactError(
                f"required local dependency {module_name!r} is missing or unresolved"
            )
        path = candidates[0]
    if path.suffix != ".py":
        raise StopArtifactError(
            f"required local dependency {module_name!r} does not resolve to Python source"
        )
    if not path.is_file():
        raise StopArtifactError(f"required local dependency source is missing: {path}")
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise StopArtifactError(
            f"required local dependency {module_name!r} resolves outside the repository"
        ) from exc
    if path.stat().st_size == 0:
        raise StopArtifactError(f"required local dependency source is empty: {path}")
    return path


def _source_key(path: Path) -> str:
    """Trả khóa nguồn tương đối theo scripts root hoặc repository root"""
    try:
        return path.relative_to(SCRIPTS_ROOT).as_posix()
    except ValueError:
        try:
            return path.relative_to(REPO_ROOT).as_posix()
        except ValueError as exc:
            raise StopArtifactError(f"unknown local module path: {path}") from exc


def _executed_module_hashes() -> dict[str, str]:
    """Băm mọi module STOP cùng dependency source quyết định capture và replay"""
    package_paths = {path.resolve() for path in SCRIPT_DIR.glob("*.py")}
    required_package_paths = {
        (SCRIPTS_ROOT / relative).resolve() for relative in STOP_PACKAGE_REQUIRED_FILES
    }
    missing_package_paths = required_package_paths - package_paths
    if missing_package_paths:
        missing = sorted(_source_key(path) for path in missing_package_paths)
        raise StopArtifactError(f"required STOP module source is missing: {missing}")

    paths = set(package_paths)
    paths.update(_source_module_path(name) for name in DEPENDENCY_MODULES)
    hashes: dict[str, str] = {}
    for path in sorted(paths):
        if path.suffix != ".py" or not path.is_file() or path.stat().st_size == 0:
            raise StopArtifactError(f"required source is missing, non-source, or empty: {path}")
        key = _source_key(path)
        hashes[key] = sha256_file(path)
    return hashes


def _verify_executed_module_hashes(manifest: StopRunManifest) -> None:
    """Xác nhận roster và bytes source trước khi reload trace hay result"""
    expected = _executed_module_hashes()
    recorded = manifest.executed_module_hashes
    missing = expected.keys() - recorded.keys()
    unknown = recorded.keys() - expected.keys()
    if missing:
        raise StopArtifactError(
            f"manifest is missing executed module hashes: {sorted(missing)}"
        )
    if unknown:
        raise StopArtifactError(
            f"manifest has unknown executed module paths: {sorted(unknown)}"
        )
    for path, digest in expected.items():
        if recorded[path] != digest:
            raise StopArtifactError(f"executed module hash mismatch for {path}")


def _git_output(arguments: list[str]) -> str:
    """Đọc stdout của một lệnh git, phân biệt cây sạch với git không đọc được"""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root(),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"git {' '.join(arguments)} is unavailable: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout


def execution_provenance() -> dict[str, Any]:
    """Ghi revision, dirty state thật và hash của đúng các module đã thực thi"""
    revision = _git_output(["rev-parse", "HEAD"]).strip()
    if not revision:
        raise RuntimeError("run provenance requires a git revision")
    porcelain = _git_output(["status", "--porcelain", "--untracked-files=all"])
    return {
        "execution_revision": revision,
        "execution_dirty": bool(porcelain.strip()),
        "executed_module_hashes": _executed_module_hashes(),
    }


def write_manifest(run_dir: Path, manifest: StopRunManifest) -> None:
    """Cập nhật manifest của run theo cách atomic"""
    _atomic_write_json(Path(run_dir) / MANIFEST_FILE, manifest.model_dump(mode="json"))


def read_manifest(run_dir: Path) -> StopRunManifest:
    """Đọc và validate manifest của run"""
    path = Path(run_dir) / MANIFEST_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StopArtifactError(f"{path}: invalid JSON at line {exc.lineno}") from exc
    except OSError as exc:
        raise StopArtifactError(f"{path}: cannot read the manifest: {exc}") from exc
    try:
        return StopRunManifest.model_validate(payload)
    except ValueError as exc:
        raise StopArtifactError(f"{path}: invalid manifest: {exc}") from exc


def write_input_traces(run_dir: Path, traces: list[StopInputTrace]) -> None:
    """Ghi toàn bộ input trace theo cách atomic"""
    path = Path(run_dir) / TRACES_FILE
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for trace in traces:
            target.write(json.dumps(trace.model_dump(mode="json"), ensure_ascii=False))
            target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def read_input_traces(run_dir: Path) -> list[StopInputTrace]:
    """Đọc và validate input trace, từ chối trace trùng hoặc thiếu"""
    path = Path(run_dir) / TRACES_FILE
    if not path.exists():
        raise StopArtifactError(f"{path}: required input trace file is missing")
    traces: list[StopInputTrace] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise StopArtifactError(f"{path}: blank line at {line_number}")
            try:
                trace = StopInputTrace.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise StopArtifactError(f"{path}: invalid trace at line {line_number}") from exc
            if trace.input_trace_id in seen:
                raise StopArtifactError(
                    f"{path}: duplicate input_trace_id {trace.input_trace_id}"
                )
            seen.add(trace.input_trace_id)
            traces.append(trace)
    return traces


def _result_path(run_dir: Path, policy: str) -> Path:
    return Path(run_dir) / f"results_{policy}.jsonl"


def append_result(run_dir: Path, result: StopResult) -> None:
    """Thêm một result rồi flush xuống đĩa để run bị ngắt vẫn kiểm tra được"""
    path = _result_path(run_dir, result.policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as target:
        target.write(line)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def build_stop_manifest(
    plan: StopPlan,
    *,
    policy_config: dict[str, dict[str, Any]],
    run_id: str,
    status: str = "prepared",
) -> StopRunManifest:
    """Dựng manifest của một run mới từ plan đã validate"""
    provenance = execution_provenance()
    return StopRunManifest(
        schema_version=STOP_POLICY_SCHEMA_VERSION,
        experiment="stop-policy",
        run_purpose=plan.run_purpose,
        run_id=run_id,
        started_at=utc_now(),
        ended_at=None,
        status=status,
        variant=plan.variant,
        split=plan.split,
        policies=list(plan.policies),
        repeats=plan.repeats,
        cases_source=plan.cases_source,
        snapshot_source=plan.snapshot_source,
        corpus_sha256=plan.corpus_sha256,
        exclusions=list(plan.exclusions),
        trials=list(plan.trials),
        input_trace_ids=[trace.input_trace_id for trace in plan.traces],
        readiness=plan.readiness,
        execution_config=None,
        execution_config_reason=PREPARED_CONFIG_REASON,
        policy_config=policy_config,
        expected_trial_count=plan.expected_trial_count,
        expected_policy_result_count=plan.expected_policy_result_count,
        execution_revision=provenance["execution_revision"],
        execution_dirty=provenance["execution_dirty"],
        executed_module_hashes=provenance["executed_module_hashes"],
    )


def create_stop_run(
    run_dir: Path,
    manifest: StopRunManifest,
    traces: list[StopInputTrace],
    report_text: str,
) -> Path:
    """Tạo run mới chỉ với trace, manifest và report ở trạng thái prepared"""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise ValueError(f"{run_dir}: destination already exists")
    run_dir.mkdir(parents=True)
    write_input_traces(run_dir, traces)
    write_manifest(run_dir, manifest)
    (run_dir / REPORT_FILE).write_text(report_text, encoding="utf-8")
    return run_dir


def prepare_run(plan: StopPlan, output_dir: Path) -> StopRunManifest:
    """Materialize một run prepared hoàn toàn offline, không tạo model client

    Params:
    - plan: plan đã validate của một split
    - output_dir: đích mới, bị từ chối nếu đã tồn tại
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"{output_dir}: destination already exists")
    manifest = build_stop_manifest(
        plan,
        policy_config={policy: {"status": "not_initialized"} for policy in plan.policies},
        run_id=uuid.uuid4().hex[:12],
        status="prepared",
    )
    create_stop_run(
        output_dir,
        manifest,
        plan.traces,
        render_stop_report(manifest, plan.traces, [], None),
    )
    return manifest


def _verified_source(reference: Any) -> Path:
    try:
        return resolve_source_file(reference)
    except (OSError, ValueError) as exc:
        raise StopArtifactError(f"invalid canonical source reference: {exc}") from exc


def _verify_schedule(
    manifest: StopRunManifest,
    cases: list[StopPolicyCase],
    snapshot: SeedSnapshot,
    cases_path: Path,
    snapshot_path: Path,
) -> None:
    """Dựng lại lịch và exclusion từ nguồn canonical rồi đối chiếu manifest"""
    try:
        if manifest.run_purpose == "later_hop_capture":
            from stop_policy_eval.schedule import build_hop0_capture_plan

            if len(manifest.policies) != 1:
                raise ValueError("later-hop source runs require exactly one policy")
            plan = build_hop0_capture_plan(
                cases,
                snapshot,
                None,
                cases_path=cases_path,
                snapshot_path=snapshot_path,
                split=manifest.split,
                policy=manifest.policies[0],
                variant=manifest.variant,
            )
        else:
            plan = build_stop_plan(
                cases,
                snapshot,
                None,
                cases_path=cases_path,
                snapshot_path=snapshot_path,
                split=manifest.split,
                policies=list(manifest.policies),
                repeats=manifest.repeats,
                variant=manifest.variant,
            )
    except ValueError as exc:
        raise StopArtifactError(f"run schedule cannot be reproduced: {exc}") from exc

    if [trial.model_dump(mode="json") for trial in plan.trials] != [
        trial.model_dump(mode="json") for trial in manifest.trials
    ]:
        raise StopArtifactError("manifest trials differ from the recomputed schedule")
    if [item.model_dump(mode="json") for item in plan.exclusions] != [
        item.model_dump(mode="json") for item in manifest.exclusions
    ]:
        raise StopArtifactError("manifest exclusions differ from the recomputed exclusions")
    if plan.expected_trial_count != manifest.expected_trial_count:
        raise StopArtifactError("manifest expected trial count differs from the schedule")
    if plan.expected_policy_result_count != manifest.expected_policy_result_count:
        raise StopArtifactError("manifest expected result count differs from the schedule")
    if [trace.input_trace_id for trace in plan.traces] != list(manifest.input_trace_ids):
        raise StopArtifactError("manifest trace ids differ from the recomputed traces")
    scheduled = {trial.case_id for trial in manifest.trials}
    if scheduled != set(plan.selected_case_ids):
        raise StopArtifactError("manifest trials do not cover the recomputed case selection")


def _verify_traces(
    manifest: StopRunManifest,
    traces: list[StopInputTrace],
    cases: list[StopPolicyCase],
) -> None:
    """Đối chiếu trace đã lưu với bản render lại từ case canonical"""
    case_by_id = {case.case_id: case for case in cases}
    if [trace.input_trace_id for trace in traces] != list(manifest.input_trace_ids):
        raise StopArtifactError("stored input traces differ from the manifest trace ids")
    fingerprint_by_trace = {
        trace.input_trace_id: trace.input_fingerprint for trace in traces
    }
    for trace in traces:
        case = case_by_id.get(trace.case_id)
        if case is None:
            raise StopArtifactError(f"trace {trace.input_trace_id} references an unknown case")
        expected = render_baseline_trace(case, manifest.variant)
        if expected.model_dump(mode="json") != trace.model_dump(mode="json"):
            raise StopArtifactError(
                f"trace {trace.input_trace_id} is not the reproducible baseline request"
            )
    for trial in manifest.trials:
        if trial.input_trace_id not in fingerprint_by_trace:
            raise StopArtifactError(f"trial {trial.trial_id} references an unknown input trace")
        if trial.case_id not in case_by_id:
            raise StopArtifactError(f"trial {trial.trial_id} references an unknown case")
        if trial.input_fingerprint != fingerprint_by_trace[trial.input_trace_id]:
            raise StopArtifactError(f"trial {trial.trial_id} changes the input fingerprint")


def read_results(
    run_dir: Path,
    manifest: StopRunManifest,
    cases: list[StopPolicyCase],
    traces: list[StopInputTrace],
) -> list[StopResult]:
    """Đọc result compact, chấm lại từ nhãn và đối chiếu mọi binding của run"""
    case_by_id = {case.case_id: case for case in cases}
    trial_by_id = {trial.trial_id: trial for trial in manifest.trials}
    trace_by_id = {trace.input_trace_id: trace for trace in traces}
    result_paths = {
        path.name[len("results_") : -len(".jsonl")]: path
        for path in Path(run_dir).glob("results_*.jsonl")
        if path.is_file()
    }
    unlisted_policies = result_paths.keys() - set(manifest.policies)
    if unlisted_policies:
        names = [result_paths[policy].name for policy in sorted(unlisted_policies)]
        raise StopArtifactError(f"unlisted policy result file(s): {names}")

    records: list[StopResult] = []
    for policy in manifest.policies:
        path = result_paths.get(policy)
        if path is None:
            continue
        seen: set[str] = set()
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise StopArtifactError(f"{path}: blank line at {line_number}")
                try:
                    record = StopResult.model_validate(json.loads(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise StopArtifactError(
                        f"{path}: invalid result at line {line_number}"
                    ) from exc
                if record.policy != policy:
                    raise StopArtifactError(
                        f"{path}: line {line_number} has policy {record.policy!r}"
                    )
                if record.run_id != manifest.run_id:
                    raise StopArtifactError(f"{path}: line {line_number} has a different run_id")
                trial = trial_by_id.get(record.trial_id)
                if trial is None:
                    raise StopArtifactError(
                        f"{path}: line {line_number} references an unscheduled trial"
                    )
                if record.input_trace_id != trial.input_trace_id:
                    raise StopArtifactError(
                        f"{path}: line {line_number} references a different input trace"
                    )
                trace = trace_by_id[trial.input_trace_id]
                case = case_by_id[trial.case_id]
                expected_fingerprint = request_fingerprint(
                    trace, policy, manifest.execution_config
                )
                if record.request_fingerprint != expected_fingerprint:
                    raise StopArtifactError(
                        f"{path}: line {line_number} changed the request fingerprint"
                    )
                score = derive_result_score(case, list(trace.candidates), record.outcome)
                for field, value in score.items():
                    if getattr(record, field) != value:
                        raise StopArtifactError(
                            f"{path}: line {line_number} has a stale {field}: "
                            f"stored {getattr(record, field)!r}, derived {value!r}"
                        )
                if record.trial_id in seen:
                    raise StopArtifactError(
                        f"{path}: duplicate result for trial {record.trial_id}"
                    )
                seen.add(record.trial_id)
                records.append(record)
    return records


def run_status(manifest: StopRunManifest, results: list[StopResult]) -> str:
    """Suy ra trạng thái run từ các slot đã ghi"""
    scheduled = {
        (policy, trial.trial_id) for policy in manifest.policies for trial in manifest.trials
    }
    completed = {(result.policy, result.trial_id) for result in results}
    if completed != scheduled:
        return "incomplete"
    if any(result.outcome.status == "error" for result in results):
        return "completed_with_errors"
    return "complete"


def load_run(
    run_dir: Path,
) -> tuple[StopRunManifest, list[StopPolicyCase], list[StopInputTrace], list[StopResult]]:
    """Reload run từ nguồn canonical và kiểm lại schedule, trace và result"""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    _verify_executed_module_hashes(manifest)
    cases_path = _verified_source(manifest.cases_source)
    snapshot_path = _verified_source(manifest.snapshot_source)
    try:
        cases = read_stop_cases(cases_path)
    except ValueError as exc:
        raise StopArtifactError(f"invalid case file: {exc}") from exc
    snapshot = read_snapshot(snapshot_path)
    for case in cases:
        if case.snapshot_id != snapshot.snapshot_id:
            raise StopArtifactError(f"{case.case_id}: case snapshot id differs from the snapshot")
        if case.source_dataset_hash != manifest.corpus_sha256:
            raise StopArtifactError(
                f"{case.case_id}: case source dataset hash differs from the manifest corpus hash"
            )
    _verify_schedule(manifest, cases, snapshot, cases_path, snapshot_path)
    traces = read_input_traces(run_dir)
    _verify_traces(manifest, traces, cases)
    results = read_results(run_dir, manifest, cases, traces)
    return manifest, cases, traces, results


def _default_client_factory() -> Any:
    """Dựng client gate production, chỉ gọi khi policy llm được yêu cầu"""
    from policies.llm import create_client

    return create_client()


def execute_run(
    run_dir: Path,
    *,
    client_factory: Callable[[], Any] | None = None,
) -> None:
    """Chạy schedule đã prepared: ghi cấu hình hiệu lực rồi append từng result

    Params:
    - run_dir: run đã prepared, chưa có result nào
    - client_factory: factory client gate, mặc định là production factory
    """
    manifest, cases, traces, results = load_run(run_dir)
    if results:
        raise StopArtifactError(
            "the run already contains result records; resuming inference is not supported"
        )
    if manifest.status not in {"prepared", "running"}:
        raise StopArtifactError(f"run status {manifest.status!r} cannot be executed")

    clients: dict[str, Any] = {}
    policy_config = dict(manifest.policy_config)
    execution_config = manifest.execution_config
    if "llm" in manifest.policies:
        factory = client_factory or _default_client_factory
        client = factory()
        execution_config = effective_execution_config(client)
        policy_config["llm"] = {
            **execution_config.model_dump(mode="json"),
            "status": "initialized",
        }
        clients["llm"] = client
    if "first" in manifest.policies:
        policy_config["first"] = {
            "status": "deterministic",
            "model_calls": 0,
            "description": "always follows the first presented candidate",
        }
    running = manifest.model_copy(
        update={
            "status": "running",
            "policy_config": policy_config,
            "execution_config": execution_config,
            "execution_config_reason": (
                manifest.execution_config_reason if execution_config is None else None
            ),
        }
    )
    write_manifest(run_dir, running)

    case_by_id = {case.case_id: case for case in cases}
    trace_by_id = {trace.input_trace_id: trace for trace in traces}
    for trial in running.trials:
        trace = trace_by_id[trial.input_trace_id]
        case = case_by_id[trial.case_id]
        for policy in running.policies:
            outcome = execute_stop_request(
                trace,
                policy=policy,
                client=clients.get(policy),
            )
            score = derive_result_score(case, list(trace.candidates), outcome)
            append_result(
                run_dir,
                StopResult(
                    run_id=running.run_id,
                    trial_id=trial.trial_id,
                    policy=policy,
                    input_trace_id=trial.input_trace_id,
                    request_fingerprint=request_fingerprint(
                        trace, policy, execution_config
                    ),
                    outcome=outcome,
                    **score,
                ),
            )


def _save_summary(
    run_dir: Path,
    manifest: StopRunManifest,
    cases: list[StopPolicyCase],
    traces: list[StopInputTrace],
    results: list[StopResult],
) -> StopSummary:
    summary = summarize_stop_run(manifest, cases, results)
    _atomic_write_json(Path(run_dir) / SUMMARY_FILE, summary.model_dump(mode="json"))
    (Path(run_dir) / REPORT_FILE).write_text(
        render_stop_report(manifest, traces, results, summary), encoding="utf-8"
    )
    write_manifest(run_dir, manifest)
    return summary


def _final_status(manifest: StopRunManifest, results: list[StopResult]) -> str:
    if manifest.status == "prepared" and not results:
        return "prepared"
    return run_status(manifest, results)


def finalize_run(run_dir: Path) -> str | None:
    """Reload artifact rồi ghi trạng thái cuối suy ra từ coverage đã lưu"""
    try:
        manifest, cases, traces, results = load_run(run_dir)
        status = _final_status(manifest, results)
        updated = manifest.model_copy(
            update={"status": status, "ended_at": manifest.ended_at or utc_now()}
        )
        if updated.run_purpose == "later_hop_capture":
            write_manifest(run_dir, updated)
            (Path(run_dir) / REPORT_FILE).write_text(
                render_source_capture_report(updated, results),
                encoding="utf-8",
            )
            return status
        _save_summary(run_dir, updated, cases, traces, results)
        return status
    except Exception as exc:  # noqa: BLE001
        print(f"Could not finalize run: {sanitize_error(exc)}", file=sys.stderr)
        return None


def summarize_run(run_dir: Path) -> str:
    """Tính lại summary và report hoàn toàn offline, không cần model hay debug file"""
    manifest, cases, traces, results = load_run(run_dir)
    if manifest.run_purpose == "later_hop_capture":
        raise StopArtifactError(
            "later-hop source runs are not scored evaluation runs; use export-later-hop"
        )
    status = _final_status(manifest, results)
    updated = manifest.model_copy(
        update={"status": status, "ended_at": manifest.ended_at or utc_now()}
    )
    _save_summary(run_dir, updated, cases, traces, results)
    return status


__all__ = [
    "EXECUTED_MODULES",
    "MANIFEST_FILE",
    "REPORT_FILE",
    "SUMMARY_FILE",
    "TRACES_FILE",
    "StopArtifactError",
    "append_result",
    "build_stop_manifest",
    "create_stop_run",
    "execute_run",
    "execution_provenance",
    "finalize_run",
    "load_run",
    "prepare_run",
    "read_input_traces",
    "read_manifest",
    "read_results",
    "run_status",
    "summarize_run",
    "write_input_traces",
    "write_manifest",
]
