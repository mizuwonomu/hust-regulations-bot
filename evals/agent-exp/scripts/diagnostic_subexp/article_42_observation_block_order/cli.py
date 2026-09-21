"""Entrypoint cho prepare, run và summarize của intervention B.

CLI là API công khai duy nhất mà orchestrator dual-suite dùng để điều khiển
sub-experiment Article 42 observation-block-order.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = next(
    candidate
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents)
    if (candidate / "diagnostic_subexp").is_dir()
)
REPO_ROOT = next(
    candidate for candidate in SCRIPTS_ROOT.parents if (candidate / "pyproject.toml").is_file()
)
if __package__ in {None, ""}:
    for bootstrap_path in (REPO_ROOT, SCRIPTS_ROOT):
        value = str(bootstrap_path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

from artifacts import read_snapshot
from contracts import GateCase
from run_experiments import _sanitize_error, _utc_now
from seed_cases import read_cases

from diagnostic_subexp.article_42_observation_block_order.artifacts import (
    ObservationArtifactError,
    append_diagnostic_result,
    build_observation_manifest,
    create_observation_run,
    derive_result_score,
    finalize_observation_run,
    load_observation_run,
    run_status,
    write_manifest,
)
from diagnostic_subexp.article_42_observation_block_order.contracts import (
    ObservationInputTrace,
    ObservationManifest,
    ObservationPlan,
)
from diagnostic_subexp.article_42_observation_block_order.metrics import (
    summarize_observation_run,
)
from diagnostic_subexp.article_42_observation_block_order.report import render_observation_report
from diagnostic_subexp.article_42_observation_block_order.schedule import (
    plan_observation_run,
    read_observation_spec,
    resolve_observation_sources,
)
from diagnostic_subexp.shared.contracts import DiagnosticResult, PrepareOutcome
from diagnostic_subexp.shared.execution import (
    effective_client_config,
    execute_diagnostic_request,
    request_from_trace,
    verify_execution_config,
)


def validate_run(
    spec_path: Path | None,
    policies: list[str],
) -> tuple[dict[str, Path], ObservationPlan]:
    """Kiểm tra definition B và dựng plan mà không ghi gì xuống đĩa."""
    resolved_spec_path = Path(spec_path) if spec_path is not None else None
    spec = read_observation_spec(resolved_spec_path)
    sources = resolve_observation_sources(spec)
    cases = read_cases(sources["case"])
    snapshot = read_snapshot(sources["snapshot"])
    plan = plan_observation_run(spec, cases, snapshot, list(policies))
    return sources, plan


def materialize_run(
    validated: tuple[dict[str, Path], ObservationPlan],
    output_dir: Path,
    policies: list[str],
) -> PrepareOutcome:
    """Ghi một run B prepared từ plan đã validate, trừ khi path đã tồn tại."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        # Result đang tồn tại là boundary chỉ đọc; không mở bất kỳ file con nào
        return PrepareOutcome(status="existing-result", run_dir=output_dir)
    _, plan = validated
    manifest = build_observation_manifest(
        plan=plan,
        policy_config={policy: {"status": "not_initialized"} for policy in policies},
        run_id=uuid.uuid4().hex[:12],
        status="prepared",
    )
    run_dir = create_observation_run(output_dir, manifest, plan.traces)
    report = render_observation_report(manifest, plan.traces, [], None)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return PrepareOutcome(status="created", run_dir=run_dir, manifest=manifest, plan=plan)


def prepare_run(
    spec_path: Path | None,
    output_dir: Path,
    policies: list[str],
) -> PrepareOutcome:
    """Kiểm tra definition B và dựng run mới mà không tạo model client."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        return PrepareOutcome(status="existing-result", run_dir=output_dir)
    return materialize_run(validate_run(spec_path, policies), output_dir, list(policies))


def _build_clients(manifest: ObservationManifest) -> dict[str, object]:
    clients: dict[str, object] = {}
    if "llm" not in manifest.policies:
        return clients
    from policies.llm import create_client

    client = create_client()
    verify_execution_config(manifest.execution_config, client)
    clients["llm"] = client
    return clients


def execute_schedule(
    run_dir: Path,
    manifest: ObservationManifest,
    cases: list[GateCase],
    traces: list[ObservationInputTrace],
    clients: dict[str, object],
) -> None:
    """Chạy lịch đã lưu và append từng result ngay sau mỗi invocation."""
    case_by_id = {case.case_id: case for case in cases}
    trace_by_id = {trace.input_trace_id: trace for trace in traces}
    for trial in manifest.trials:
        trace = trace_by_id[trial.input_trace_id]
        case = case_by_id[trial.case_id]
        request = request_from_trace(trace, case)
        for policy in manifest.policies:
            outcome = execute_diagnostic_request(
                request,
                policy=policy,
                client=clients.get(policy),
            )
            score = derive_result_score(case, trial.presented_candidates, outcome)
            append_diagnostic_result(
                run_dir,
                DiagnosticResult(
                    run_id=manifest.run_id,
                    trial_id=trial.trial_id,
                    policy=policy,
                    input_trace_id=trial.input_trace_id,
                    request_fingerprint=trial.request_fingerprint,
                    outcome=outcome,
                    **score,
                ),
            )


def _execute_run(run_dir: Path, manifest: ObservationManifest) -> None:
    _, cases, traces, results = load_observation_run(run_dir)
    if results:
        raise ObservationArtifactError(
            "the run already contains result records; resuming inference is not supported"
        )
    if manifest.status not in {"prepared", "running"}:
        raise ObservationArtifactError(f"run status {manifest.status!r} cannot be executed")

    clients = _build_clients(manifest)
    policy_config = dict(manifest.policy_config)
    if "llm" in clients:
        policy_config["llm"] = effective_client_config(clients["llm"])
    running = manifest.model_copy(update={"status": "running", "policy_config": policy_config})
    write_manifest(run_dir, running)
    execute_schedule(run_dir, running, cases, traces, clients)


def _save_summary(run_dir, manifest, cases, traces, results):
    summary = summarize_observation_run(manifest, cases, traces, results)
    report = render_observation_report(manifest, traces, results, summary)
    finalize_observation_run(run_dir, manifest, summary, report)


def finalize_run(run_dir: Path) -> str | None:
    """Reload artifact và ghi trạng thái cuối suy ra từ coverage đã lưu."""
    try:
        manifest, cases, traces, results = load_observation_run(run_dir)
        status = run_status(manifest, results)
        updated = manifest.model_copy(update={"status": status, "ended_at": _utc_now()})
        _save_summary(run_dir, updated, cases, traces, results)
        return status
    except Exception as exc:  # noqa: BLE001
        print(f"Could not finalize run: {_sanitize_error(exc)}", file=sys.stderr)
        return None


def summarize_run(run_dir: Path) -> str:
    """Tính lại summary và report của B hoàn toàn offline."""
    manifest, cases, traces, results = load_observation_run(run_dir)
    status = run_status(manifest, results) if manifest.status != "prepared" else "prepared"
    updated = manifest.model_copy(update={"status": status, "ended_at": manifest.ended_at or _utc_now()})
    _save_summary(run_dir, updated, cases, traces, results)
    return status


def execute_run(run_dir: Path) -> None:
    """Chạy schedule của run B đã prepared, set running rồi append từng result."""
    manifest = load_observation_run(run_dir)[0]
    _execute_run(run_dir, manifest)


def _run_command(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    manifest = load_observation_run(run_dir)[0]
    interrupted = False
    failure: Exception | None = None
    try:
        _execute_run(run_dir, manifest)
    except KeyboardInterrupt:
        interrupted = True
        print("Run interrupted", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        failure = exc
        print(f"Run failed: {_sanitize_error(exc)}", file=sys.stderr)
    status = finalize_run(run_dir)
    if interrupted:
        return 130
    if failure is not None or status is None:
        return 1
    return 0 if status == "complete" else 1


def _summarize_command(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    status = summarize_run(run_dir)
    print(f"Summary written to {run_dir / 'summary.json'}")
    print(f"Report written to {run_dir / 'report.md'}")
    return 0 if status == "complete" else 1


def _build_parser() -> argparse.ArgumentParser:
    """Tạo parser ba phase cho intervention B."""
    parser = argparse.ArgumentParser(description="Article 42 observation-block-order diagnostic runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="validate the embedded definition and materialize an unstarted run",
    )
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--policies", nargs="+", choices=["first", "llm"], default=["first", "llm"])

    run = subparsers.add_parser("run", help="execute a prepared diagnostic schedule")
    run.add_argument("--run-dir", type=Path, required=True)

    summarize = subparsers.add_parser(
        "summarize",
        help="recompute the summary and report offline",
    )
    summarize.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Điều phối một command B và trả exit code đã quy ước."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            outcome = prepare_run(None, Path(args.output_dir), list(args.policies))
            if outcome.status == "existing-result":
                print(f"Existing result at {outcome.run_dir}; left untouched")
                return 0
            print(f"Run directory: {outcome.run_dir}")
            print(
                f"Scheduled {outcome.plan.expected_trial_count} trials per policy configuration "
                f"({outcome.plan.expected_policy_result_count} result slots)"
            )
            return 0
        if args.command == "run":
            return _run_command(args)
        if args.command == "summarize":
            return _summarize_command(args)
    except KeyboardInterrupt:
        return 130
    except (ObservationArtifactError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


__all__ = [
    "execute_run",
    "execute_schedule",
    "finalize_run",
    "main",
    "materialize_run",
    "prepare_run",
    "summarize_run",
    "validate_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
