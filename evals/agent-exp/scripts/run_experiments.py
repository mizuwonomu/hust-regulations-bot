"""CLI for preparing and replaying the initial citation-gate experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import openai

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from artifacts import (
    ArtifactCorruptionError,
    append_debug_record,
    append_policy_log,
    append_result,
    create_run,
    finalize_run,
    load_run,
    read_snapshot,
    sha256_file,
    write_manifest,
    write_snapshot,
)
from contracts import (
    DecisionOutcome,
    GateCase,
    RunManifest,
    SourceFile,
    Trial,
    TrialError,
)
from metrics import score_result, summarize
from seed_cases import (
    import_seeds,
    policy_input,
    prepare_cases,
    read_cases,
    validate_replay,
    write_cases,
)

from src.rag.agent.schema import Decision


class PolicyInputError(ValueError):
    """Signal an invalid policy decision that must not become STOP."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git_value(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if completed.returncode not in {0, 1}:
        return None
    return completed.returncode == 1


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _code_hashes() -> dict[str, str]:
    paths = [
        SCRIPT_DIR / "contracts.py",
        SCRIPT_DIR / "seed_cases.py",
        SCRIPT_DIR / "policies" / "first.py",
        SCRIPT_DIR / "policies" / "llm.py",
        SCRIPT_DIR / "artifacts.py",
        SCRIPT_DIR / "metrics.py",
        SCRIPT_DIR / "run_experiments.py",
        REPO_ROOT / "src/rag/agent/gate.py",
        REPO_ROOT / "src/rag/agent/prompt.py",
        REPO_ROOT / "src/rag/agent/llm_client.py",
    ]
    return {
        _relative_path(path): sha256_file(path)
        for path in paths
        if path.exists()
    }


def _provenance() -> dict[str, Any]:
    revision = _git_value(["rev-parse", "HEAD"])
    dirty = _git_dirty()
    return {
        "source_revision": revision,
        "source_revision_reason": None if revision else "git revision unavailable",
        "working_tree_modified": dirty,
        "working_tree_modified_reason": None if dirty is not None else "git status unavailable",
        "code_hashes": _code_hashes(),
        "python_version": sys.version.split()[0],
    }


def _initial_policy_config(policies: list[str]) -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {}
    if "first" in policies:
        config["first"] = {
            "model_calls": 0,
            "description": "always follow candidates[0]",
        }
    if "llm" in policies:
        config["llm"] = {
            "model_alias": None,
            "actual_model": None,
            "actual_model_reason": "client is initialized after the schedule is saved",
            "temperature": None,
            "sampling": {"temperature": None},
            "thinking": None,
            "token_limit": None,
            "timeout_seconds": None,
            "retry_settings": None,
            "prompt_file_hash": sha256_file(REPO_ROOT / "src/rag/agent/prompt.py"),
            "grammar_source_hash": sha256_file(REPO_ROOT / "src/rag/agent/gate.py"),
            "gate_file_hash": sha256_file(REPO_ROOT / "src/rag/agent/gate.py"),
        }
    return config


def plan_trials(
    cases: list[GateCase],
    *,
    repeats: int,
    seed_order_by_case: dict[str, list[int]] | None = None,
) -> list[Trial]:
    """Create shared original-order trials for approved semantic cases."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")

    seen_case_ids: set[str] = set()
    trials: list[Trial] = []
    for case in cases:
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)
        if case.label_status != "approved" or case.expected_action not in {"follow", "stop"}:
            continue
        if not case.candidates:
            raise ValueError(f"approved semantic case has no candidates: {case.case_id}")
        for repeat_id in range(repeats):
            trial_id = f"{case.case_id}:r{repeat_id}"
            trials.append(
                Trial(
                    trial_id=trial_id,
                    case_id=case.case_id,
                    repeat_id=repeat_id,
                    condition="original",
                    permutation_id="original",
                    seed_order=list((seed_order_by_case or {}).get(case.case_id, [])),
                    candidate_order=list(case.candidates),
                    observation_hash=case.observation_hash,
                )
            )
    return trials


def _sanitize_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if not message:
        message = type(exc).__name__
    return message[:500]


def _classify_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    # Ưu tiên timeout trước transport vì timeout cũng có thể thuộc nhóm lỗi kết nối
    if (
        isinstance(
            exc,
            (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException, openai.APITimeoutError),
        )
        or "timeout" in name
        or "timed out" in message
    ):
        return "timeout"
    if isinstance(exc, (httpx.TransportError, openai.APIConnectionError, ConnectionError, OSError)):
        return "transport"
    if "candidate" in message or "outside" in message or "positive integer" in message:
        return "invalid_candidate"
    if any(
        marker in name or marker in message
        for marker in ("transport", "requesterror", "connection refused", "connection reset")
    ):
        return "transport"
    if isinstance(exc, (json.JSONDecodeError,)) or any(
        marker in name or marker in message
        for marker in ("validation", "schema", "json", "malformed", "decision")
    ):
        return "schema"
    return "unexpected"


def _outcome_from_exception(exc: Exception, latency_ms: float) -> DecisionOutcome:
    return DecisionOutcome(
        status="error",
        decision=None,
        error=TrialError(category=_classify_exception(exc), message=_sanitize_error(exc)),
        latency_ms=latency_ms,
        usage=None,
    )


def _validate_decision(decision: Any, candidates: list[int]) -> Decision:
    if not isinstance(decision, Decision):
        raise TypeError("policy returned a non-Decision value")
    if not decision.stop and decision.dieu not in candidates:
        raise PolicyInputError(
            f"policy selected candidate {decision.dieu!r} outside {candidates!r}"
        )
    return decision


def _execute_trial(
    run_dir: Path,
    manifest: RunManifest,
    case: GateCase,
    trial: Trial,
    policy: str,
    decide: Callable[..., Decision],
    client: Any | None = None,
) -> Any:
    started = time.perf_counter()
    input_record = policy_input(case)
    raw_decision: Decision | None = None
    try:
        if policy == "llm":
            raw_decision = decide(
                input_record.question,
                input_record.observation,
                input_record.candidates,
                client=client,
            )
        else:
            raw_decision = decide(
                input_record.question,
                input_record.observation,
                input_record.candidates,
            )
        decision = _validate_decision(raw_decision, input_record.candidates)
        outcome = DecisionOutcome(
            status="ok",
            decision=decision,
            error=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=None,
        )
    except Exception as exc:  # noqa: BLE001
        outcome = _outcome_from_exception(
            exc,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    result = score_result(manifest.run_id, policy, trial, case, outcome)
    append_result(run_dir, result)
    append_debug_record(
        run_dir,
        policy,
        {
            "run_id": manifest.run_id,
            "trial_id": trial.trial_id,
            "case_id": case.case_id,
            "question": input_record.question,
            "observation": input_record.observation,
            "candidate_order": input_record.candidates,
            "decision": raw_decision.model_dump(mode="json") if raw_decision is not None else None,
            "outcome": outcome.model_dump(mode="json"),
        },
    )
    append_policy_log(
        run_dir,
        policy,
        f"trial_id={trial.trial_id} status={outcome.status} latency_ms={outcome.latency_ms:.3f}",
    )
    return result


def _build_manifest(
    *,
    policies: list[str],
    trials: list[Trial],
    exclusions: list[Any],
    snapshot_path: Path,
    cases_path: Path,
) -> RunManifest:
    return RunManifest(
        schema_version=1,
        run_id=uuid.uuid4().hex[:12],
        experiment="initial-selection",
        started_at=_utc_now(),
        ended_at=None,
        status="running",
        policies=policies,
        trials=trials,
        cases_source=SourceFile(path=str(cases_path), sha256=sha256_file(cases_path)),
        snapshot_source=SourceFile(path=str(snapshot_path), sha256=sha256_file(snapshot_path)),
        exclusions=exclusions,
        provenance=_provenance(),
        policy_config=_initial_policy_config(policies),
    )


def _manifest_status(manifest: RunManifest, results: list[Any]) -> str:
    scheduled = {(policy, trial.trial_id) for policy in manifest.policies for trial in manifest.trials}
    completed = {(result.policy, result.trial.trial_id) for result in results}
    if completed != scheduled:
        return "incomplete"
    if any(result.outcome.status == "error" for result in results):
        return "completed_with_errors"
    return "complete"


def _run_command(args: argparse.Namespace) -> int:
    snapshot_path = Path(args.seeds)
    cases_path = Path(args.cases)
    snapshot = read_snapshot(snapshot_path)
    snapshot_hash = sha256_file(snapshot_path)
    cases = read_cases(cases_path)
    selection = validate_replay(snapshot, cases, snapshot_hash=snapshot_hash)
    if not selection.eligible_cases:
        print("No eligible approved cases remain after replay validation", file=sys.stderr)
        return 2

    policies = list(args.policies)
    if len(set(policies)) != len(policies):
        raise ValueError("policies must not contain duplicates")
    row_by_id = {(type(row.id), row.id): row for row in snapshot.rows}
    seed_order_by_case = {
        case.case_id: list(
            range(len(row_by_id[(type(case.question_id), case.question_id)].contexts))
        )
        for case in selection.eligible_cases
    }
    trials = plan_trials(
        selection.eligible_cases,
        repeats=args.repeats,
        seed_order_by_case=seed_order_by_case,
    )
    if not trials:
        print("No scheduled trials were created", file=sys.stderr)
        return 2

    manifest = _build_manifest(
        policies=policies,
        trials=trials,
        exclusions=selection.exclusions,
        snapshot_path=snapshot_path,
        cases_path=cases_path,
    )
    run_dir = create_run(Path(args.output_root), manifest)
    print(f"Run directory: {run_dir}")
    case_by_id = {case.case_id: case for case in selection.eligible_cases}
    runner_failure: Exception | None = None
    interrupted = False
    final_status: str | None = None

    try:
        for policy in policies:
            client = None
            if policy == "first":
                from policies.first import decide
            else:
                from policies.llm import client_config, create_client, decide

                client = create_client()
                policy_config = dict(manifest.policy_config)
                policy_config["llm"] = {
                    **client_config(client),
                    "prompt_file_hash": sha256_file(REPO_ROOT / "src/rag/agent/prompt.py"),
                    "grammar_source_hash": sha256_file(REPO_ROOT / "src/rag/agent/gate.py"),
                    "gate_file_hash": sha256_file(REPO_ROOT / "src/rag/agent/gate.py"),
                }
                manifest = manifest.model_copy(update={"policy_config": policy_config})
                write_manifest(run_dir, manifest)

            for trial in trials:
                _execute_trial(
                    run_dir,
                    manifest,
                    case_by_id[trial.case_id],
                    trial,
                    policy,
                    decide,
                    client=client,
                )
    except KeyboardInterrupt:
        interrupted = True
        print("Run interrupted", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        runner_failure = exc
        print(f"Run failed before scheduling all trials: {_sanitize_error(exc)}", file=sys.stderr)
    finally:
        # Recompute completion from persisted artifacts, not from in-memory results
        try:
            saved_manifest, saved_cases, saved_results = load_run(run_dir)
            if interrupted or runner_failure is not None:
                final_status = "incomplete"
            else:
                final_status = _manifest_status(saved_manifest, saved_results)
            final_manifest = saved_manifest.model_copy(
                update={"status": final_status, "ended_at": _utc_now()}
            )
            summary = summarize(final_manifest, saved_cases, saved_results)
            finalize_run(run_dir, final_manifest, summary)
        except Exception as exc:  # noqa: BLE001
            final_status = None
            print(f"Could not finalize run: {_sanitize_error(exc)}", file=sys.stderr)

    if final_status is None:
        return 1
    if interrupted:
        return 130
    if runner_failure is not None:
        return 1
    return 0 if final_status == "complete" else 1


def _summarize_command(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    manifest, cases, results = load_run(run_dir)
    status = _manifest_status(manifest, results)
    updated_manifest = manifest.model_copy(
        update={"status": status, "ended_at": manifest.ended_at or _utc_now()}
    )
    summary = summarize(updated_manifest, cases, results)
    finalize_run(run_dir, updated_manifest, summary)
    print(f"Summary written to {run_dir / 'summary.json'}")
    return 0 if status == "complete" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initial citation-gate experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-seeds", help="import a single-pass baseline")
    import_parser.add_argument("--baseline", type=Path, required=True)
    import_parser.add_argument("--dataset", type=Path, required=True)
    import_parser.add_argument("--internal-dieu", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, required=True)

    prepare_parser = subparsers.add_parser("prepare-cases", help="prepare draft gate cases")
    prepare_parser.add_argument("--seeds", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser("run", help="execute initial-selection policies")
    run_parser.add_argument("--experiment", choices=["initial-selection"], required=True)
    run_parser.add_argument("--seeds", type=Path, required=True)
    run_parser.add_argument("--cases", type=Path, required=True)
    run_parser.add_argument("--policies", nargs="+", choices=["first", "llm"], required=True)
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("--output-root", type=Path, required=True)

    summarize_parser = subparsers.add_parser("summarize", help="recompute a saved run")
    summarize_parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one CLI command and return its documented exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "import-seeds":
            snapshot = import_seeds(args.baseline, args.dataset, args.internal_dieu)
            write_snapshot(snapshot, args.output)
            print(f"Snapshot written to {args.output}")
            return 0
        if args.command == "prepare-cases":
            snapshot = read_snapshot(args.seeds)
            snapshot_hash = sha256_file(args.seeds)
            cases = prepare_cases(snapshot, snapshot_hash=snapshot_hash)
            write_cases(cases, args.output)
            print(f"Cases written to {args.output}")
            return 0
        if args.command == "run":
            if args.repeats <= 0:
                raise ValueError("repeats must be a positive integer")
            return _run_command(args)
        if args.command == "summarize":
            return _summarize_command(args)
    except KeyboardInterrupt:
        return 130
    except (ArtifactCorruptionError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
