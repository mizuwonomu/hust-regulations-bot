"""Entrypoint công khai cho capture, review, prepare, run và summarize STOP policy

`capture-corpus` nạp env file được chỉ định trước khi dựng retrieval runtime.
Các command offline không nạp env file; retrieval và gate chỉ import theo command
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Tắt tracing trước khi import module có thể dựng client
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

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

CAPTURE_FILES: tuple[str, ...] = (
    "seeds.json",
    "cases_draft.jsonl",
    "cases_review.jsonl",
    "label_review.md",
)


def _live_capture_runtime_factory(settings):
    """Dựng retrieval runtime thật; import nặng chỉ chạy sau preflight input đích"""
    from evals.common.single_pass_retrieval import build_single_pass_runtime

    return build_single_pass_runtime(settings)


def _load_capture_env_file(path: Path) -> None:
    """Nạp env file cho capture mà không ghi đè biến đã export trong shell"""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=path, override=False)


def _capture_command(args: argparse.Namespace) -> int:
    from evals.common.single_pass_retrieval import SinglePassSettings

    from stop_policy_eval.capture import capture_corpus
    from stop_policy_eval.support import sanitize_error

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"{output_dir}: destination already exists")
    settings = SinglePassSettings.effective(rerank_ratio=args.ratio)
    try:
        outcome = capture_corpus(
            Path(args.dataset),
            Path(args.internal_dieu),
            output_dir=output_dir,
            settings=settings,
            runtime_factory=_live_capture_runtime_factory,
            before_runtime=lambda: _load_capture_env_file(args.env_file),
        )
    except (ValueError, OSError):
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Capture failed: {sanitize_error(exc)}", file=sys.stderr)
        return 1

    print(f"Capture directory: {outcome.capture_dir}")
    for name in CAPTURE_FILES:
        print(f"- {name}: {outcome.capture_dir / name}")
    print(f"Captured {len(outcome.draft_cases)} draft case(s)")
    excluded = sum(1 for case in outcome.draft_cases if not case.candidates)
    print(f"Mechanical exclusions (empty frontier): {excluded}")
    print(
        "Next action: edit cases_review.jsonl, then run "
        f"`validate-cases --seeds {outcome.capture_dir / 'seeds.json'} "
        f"--corpus {Path(args.dataset)} --cases {outcome.capture_dir / 'cases_review.jsonl'}`"
    )
    return 0


def _inspect_command(args: argparse.Namespace) -> int:
    from stop_policy_eval.capture import read_capture_manifest
    from stop_policy_eval.cases import read_stop_cases

    capture_dir = Path(args.capture_dir)
    manifest = read_capture_manifest(capture_dir)
    cases = read_stop_cases(capture_dir / "cases_review.jsonl")
    print(f"Capture directory: {capture_dir}")
    print(f"- capture status: {manifest.capture_status}")
    print(f"- rows: {manifest.row_count}")
    print(f"- snapshot: {manifest.snapshot_id}")
    print(f"- snapshot hash: {manifest.snapshot_hash}")
    for name in CAPTURE_FILES:
        print(f"- {name}: {capture_dir / name}")
    print(f"- draft cases: {len(cases)}")
    print(
        "- mechanically excluded (empty frontier): "
        f"{sum(1 for case in cases if not case.candidates)}"
    )
    print(
        "- reviewed so far (approved): "
        f"{sum(1 for case in cases if case.label_status == 'approved')}"
    )
    print("Next action: edit cases_review.jsonl, then run `validate-cases`")
    return 0


def _import_command(args: argparse.Namespace) -> int:
    from artifacts import read_snapshot

    from stop_policy_eval.artifacts import load_run
    from stop_policy_eval.cases import (
        import_later_hop_states,
        read_later_hop_states,
        read_stop_cases,
        source_file,
    )
    from stop_policy_eval.later_hop import (
        verify_export_bundle,
        verify_later_hop_source_provenance,
    )

    output_path = Path(args.output)
    if output_path.exists():
        raise ValueError(f"{output_path}: destination already exists")
    source_run_dir = Path(args.source_run_dir)
    source_manifest, source_cases, _, source_results = load_run(source_run_dir)
    cases_source = source_file(Path(args.cases))
    snapshot_source = source_file(Path(args.seeds))
    if cases_source.sha256 != source_manifest.cases_source.sha256:
        raise ValueError("base case file does not match the source run cases artifact")
    if snapshot_source.sha256 != source_manifest.snapshot_source.sha256:
        raise ValueError("seeds file does not match the source run snapshot artifact")
    states_path = Path(args.states)
    verify_export_bundle(states_path, source_run_dir)
    states = read_later_hop_states(states_path, allow_empty=True)
    if not states:
        raise ValueError(f"{states_path}: export contains no importable later-hop states")
    verify_later_hop_source_provenance(
        states,
        source_manifest,
        source_cases,
        source_results,
    )
    snapshot = read_snapshot(Path(args.seeds))
    base_cases = read_stop_cases(Path(args.cases))
    outcome = import_later_hop_states(snapshot, base_cases, states, output_path)
    print(f"Case file written to {outcome.output_path}")
    print(f"- added later-hop cases: {len(outcome.added_case_ids)}")
    for case_id in outcome.added_case_ids:
        print(f"- {case_id}")
    print(
        "Next action: review the imported draft cases in the new file, then run `validate-cases`"
    )
    return 0


def _export_later_hop_command(args: argparse.Namespace) -> int:
    from stop_policy_eval.later_hop import export_later_hop

    outcome = export_later_hop(Path(args.run_dir), Path(args.output_dir))
    print(f"Later-hop export directory: {outcome.output_dir}")
    print(f"- source run id: {outcome.source_run_id}")
    print(f"- states: {outcome.output_dir / 'later_hop_states.jsonl'} ({len(outcome.states)})")
    print(
        f"- terminal outcomes: {outcome.output_dir / 'terminal_outcomes.jsonl'} "
        f"({len(outcome.terminal_outcomes)})"
    )
    print(f"- skipped non-FOLLOW or invalid results: {outcome.skipped_results}")
    print(f"- export manifest: {outcome.output_dir / 'later_hop_export_manifest.json'}")
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    from artifacts import read_snapshot

    from stop_policy_eval.cases import read_stop_cases, read_stop_corpus, validate_reviewed_cases

    snapshot = read_snapshot(Path(args.seeds))
    corpus = read_stop_corpus(Path(args.corpus))
    cases = read_stop_cases(Path(args.cases))
    selection = validate_reviewed_cases(snapshot, corpus, cases)
    print(f"Case file validated: {Path(args.cases)}")
    print(f"- eligible cases: {len(selection.eligible_cases)}")
    counts: dict[str, int] = {}
    for exclusion in selection.exclusions:
        counts[exclusion.reason] = counts.get(exclusion.reason, 0) + 1
    rendered = ", ".join(f"{reason} {count}" for reason, count in sorted(counts.items()))
    print(f"- exclusions: {rendered or 'none'}")
    for readiness in selection.readiness:
        detail = ", ".join(
            f"{row.requirement} {row.eligible_cases}" for row in readiness.requirements
        )
        state = "complete" if readiness.complete else f"missing {readiness.missing()}"
        print(f"- split {readiness.split}: {state} ({detail})")
    print("Next action: prepare a run once a split reports complete")
    return 0


def _prepare_command(args: argparse.Namespace) -> int:
    from artifacts import read_snapshot

    from stop_policy_eval.artifacts import prepare_run
    from stop_policy_eval.cases import read_stop_cases, read_stop_corpus
    from stop_policy_eval.schedule import build_stop_plan

    cases_path = Path(args.cases)
    snapshot_path = Path(args.seeds)
    snapshot = read_snapshot(snapshot_path)
    corpus = read_stop_corpus(Path(args.corpus))
    cases = read_stop_cases(cases_path)
    plan = build_stop_plan(
        cases,
        snapshot,
        corpus,
        cases_path=cases_path,
        snapshot_path=snapshot_path,
        split=args.split,
        policies=list(args.policies),
        repeats=args.repeats,
        variant=args.variant,
    )
    manifest = prepare_run(plan, Path(args.output_dir))
    print(f"Run directory: {Path(args.output_dir)}")
    print(f"- run id: {manifest.run_id}")
    print(f"- split: {manifest.split} | variant: {manifest.variant} | repeats: {manifest.repeats}")
    print(f"- policies: {', '.join(manifest.policies)}")
    print(
        f"- scheduled trials: {manifest.expected_trial_count} | result slots: "
        f"{manifest.expected_policy_result_count}"
    )
    print(f"- exclusions: {len(manifest.exclusions)}")
    print(f"Next action: run `run --run-dir {Path(args.output_dir)}`")
    return 0


def _prepare_hop0_command(args: argparse.Namespace) -> int:
    from artifacts import read_snapshot

    from stop_policy_eval.artifacts import prepare_run
    from stop_policy_eval.cases import read_stop_cases, read_stop_corpus
    from stop_policy_eval.schedule import build_hop0_capture_plan

    cases_path = Path(args.cases)
    snapshot_path = Path(args.seeds)
    snapshot = read_snapshot(snapshot_path)
    corpus = read_stop_corpus(Path(args.corpus))
    cases = read_stop_cases(cases_path)
    plan = build_hop0_capture_plan(
        cases,
        snapshot,
        corpus,
        cases_path=cases_path,
        snapshot_path=snapshot_path,
        split=args.split,
        policy=args.policy,
        variant=args.variant,
    )
    manifest = prepare_run(plan, Path(args.output_dir))
    print(f"Source run directory: {Path(args.output_dir)}")
    print(f"- run id: {manifest.run_id}")
    print(f"- purpose: {manifest.run_purpose}")
    print(f"- split: {manifest.split} | variant: {manifest.variant}")
    print(f"- source policy: {', '.join(manifest.policies)} | repeats: {manifest.repeats}")
    print(f"- hop-0 inputs with candidates: {manifest.expected_trial_count}")
    print(f"Next action: run --run-dir {Path(args.output_dir)}, then export-later-hop")
    return 0


def _run_command(args: argparse.Namespace) -> int:
    from stop_policy_eval.artifacts import execute_run, finalize_run, load_run
    from stop_policy_eval.support import sanitize_error

    run_dir = Path(args.run_dir)
    try:
        manifest, _, _, existing = load_run(run_dir)
        if existing:
            raise ValueError(
                f"{run_dir}: the run already contains result records; rerunning is not supported"
            )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    interrupted = False
    failure: Exception | None = None
    try:
        execute_run(run_dir)
    except KeyboardInterrupt:
        interrupted = True
        print("Run interrupted", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        failure = exc
        print(f"Run failed: {sanitize_error(exc)}", file=sys.stderr)

    status = finalize_run(run_dir)
    if status is not None:
        print(f"Run status: {status}")
        print(f"Report written to {run_dir / 'report.md'}")
        if manifest.run_purpose == "later_hop_capture":
            print("Source decisions recorded; no evaluation summary was written")
        else:
            print(f"Summary written to {run_dir / 'summary.json'}")
    if interrupted:
        return 130
    if failure is not None or status is None:
        return 1
    if manifest.run_purpose == "later_hop_capture":
        print(
            "Next action: export-later-hop "
            f"--run-dir {run_dir} --output-dir <new-export-directory>"
        )
    else:
        print(f"Next action: inspect {run_dir / 'report.md'}")
    return 0 if status == "complete" else 1


def _summarize_command(args: argparse.Namespace) -> int:
    from stop_policy_eval.artifacts import summarize_run

    run_dir = Path(args.run_dir)
    status = summarize_run(run_dir)
    print(f"Run status: {status}")
    print(f"Summary written to {run_dir / 'summary.json'}")
    print(f"Report written to {run_dir / 'report.md'}")
    return 0 if status == "complete" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STOP policy evaluation runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture-corpus",
        help="run live retrieval for the corpus and publish a label review bundle",
    )
    capture.add_argument("--dataset", type=Path, required=True)
    capture.add_argument("--internal-dieu", type=Path, required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--ratio", type=float, default=None)
    capture.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="load capture environment variables from this file (default: .env)",
    )

    inspect = subparsers.add_parser(
        "inspect-capture",
        help="read a capture bundle offline and print its artifact paths",
    )
    inspect.add_argument("--capture-dir", type=Path, required=True)

    import_parser = subparsers.add_parser(
        "import-later-hop",
        help="import explicitly supplied frozen later-hop states as a new case version",
    )
    import_parser.add_argument("--seeds", type=Path, required=True)
    import_parser.add_argument("--cases", type=Path, required=True)
    import_parser.add_argument("--states", type=Path, required=True)
    import_parser.add_argument("--source-run-dir", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, required=True)

    export_later_hop = subparsers.add_parser(
        "export-later-hop",
        help="fetch one followed article from a hop-0 source run and freeze its successor state",
    )
    export_later_hop.add_argument("--run-dir", type=Path, required=True)
    export_later_hop.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate-cases",
        help="validate reviewed labels, groups and splits against the corpus",
    )
    validate.add_argument("--seeds", type=Path, required=True)
    validate.add_argument("--corpus", type=Path, required=True)
    validate.add_argument("--cases", type=Path, required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="materialize an offline, unstarted replay run for one approved split",
    )
    prepare.add_argument("--seeds", type=Path, required=True)
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--cases", type=Path, required=True)
    prepare.add_argument("--variant", choices=("baseline",), default="baseline")
    prepare.add_argument("--split", choices=["dev", "heldout"], required=True)
    prepare.add_argument("--policies", nargs="+", choices=("first", "llm"), required=True)
    prepare.add_argument("--repeats", type=int, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    prepare_hop0 = subparsers.add_parser(
        "prepare-hop0",
        help="prepare one hop-0 source run for a single follow and later-hop capture",
    )
    prepare_hop0.add_argument("--seeds", type=Path, required=True)
    prepare_hop0.add_argument("--corpus", type=Path, required=True)
    prepare_hop0.add_argument("--cases", type=Path, required=True)
    prepare_hop0.add_argument("--variant", choices=("baseline",), default="baseline")
    prepare_hop0.add_argument("--split", choices=["dev", "heldout"], required=True)
    prepare_hop0.add_argument("--policy", choices=("first", "llm"), default="llm")
    prepare_hop0.add_argument("--output-dir", type=Path, required=True)

    run = subparsers.add_parser("run", help="execute a prepared run and finalize its artifacts")
    run.add_argument("--run-dir", type=Path, required=True)

    summarize = subparsers.add_parser(
        "summarize",
        help="recompute the summary and report of a run offline",
    )
    summarize.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Điều phối một command và trả exit code đã quy ước"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "capture-corpus":
            return _capture_command(args)
        if args.command == "inspect-capture":
            return _inspect_command(args)
        if args.command == "import-later-hop":
            return _import_command(args)
        if args.command == "export-later-hop":
            return _export_later_hop_command(args)
        if args.command == "validate-cases":
            return _validate_command(args)
        if args.command == "prepare":
            return _prepare_command(args)
        if args.command == "prepare-hop0":
            return _prepare_hop0_command(args)
        if args.command == "run":
            return _run_command(args)
        if args.command == "summarize":
            return _summarize_command(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
