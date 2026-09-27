"""Prepare, execute và summarize prompt ablation tiến sĩ C.

CLI là API công khai duy nhất của sub-experiment doctoral-defense-prompt-ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
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

from run_experiments import _sanitize_error, _utc_now

from diagnostic_subexp.doctoral_defense_prompt_ablation.artifacts import (
    PromptAblationArtifactError,
    execute_prompt_ablation,
    load_prompt_ablation,
    prepare_prompt_ablation,
    run_status,
    write_manifest,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.contracts import (
    PromptAblationManifest,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.metrics import summarize_prompt_ablation
from diagnostic_subexp.doctoral_defense_prompt_ablation.report import render_prompt_ablation_report
from diagnostic_subexp.shared.execution import effective_client_config, verify_execution_config


def _build_clients(manifest: PromptAblationManifest) -> dict[str, object]:
    """Chỉ tạo client nếu policy đã lưu yêu cầu."""
    clients: dict[str, object] = {}
    if "llm" in manifest.policies:
        from policies.llm import create_client

        client = create_client()
        verify_execution_config(manifest.execution_config, client)
        clients["llm"] = client
    return clients


def _write_summary_and_report(run_dir: Path, manifest: PromptAblationManifest) -> str:
    """Reload artifact, tính summary/report và ghi lại trạng thái cuối."""
    manifest, cases, inputs, traces, results = load_prompt_ablation(run_dir)
    status = manifest.status if manifest.status == "prepared" else run_status(manifest, results)
    updated = manifest.model_copy(update={"status": status, "ended_at": manifest.ended_at or _utc_now()})
    summary = summarize_prompt_ablation(updated, cases, inputs, traces, results)
    summary["status"] = status
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        render_prompt_ablation_report(updated, traces, summary), encoding="utf-8"
    )
    write_manifest(run_dir, updated)
    return status


def _execute_and_finalize(run_dir: Path) -> str:
    """Chạy một run C đã prepared và luôn finalize, kể cả khi bị ngắt."""
    manifest, cases, inputs, traces, results = load_prompt_ablation(run_dir)
    if results:
        raise PromptAblationArtifactError("the run already contains result records")
    clients = _build_clients(manifest)
    config = dict(manifest.policy_config)
    if "llm" in clients:
        config["llm"] = effective_client_config(clients["llm"])
    running = manifest.model_copy(update={"status": "running", "policy_config": config})
    write_manifest(run_dir, running)
    try:
        execute_prompt_ablation(run_dir, running, cases, inputs, traces, clients)
    finally:
        status = _write_summary_and_report(run_dir, running)
    return status


def _build_parser() -> argparse.ArgumentParser:
    """Tạo parser ba phase cho C."""
    parser = argparse.ArgumentParser(description="Doctoral-defense prompt-ablation runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="materialize schema-v3 C evidence")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--policies", nargs="+", choices=["first", "llm"], default=["first", "llm"])
    run = subparsers.add_parser("run", help="execute a prepared C schedule")
    run.add_argument("--run-dir", type=Path, required=True)
    summarize = subparsers.add_parser("summarize", help="recompute C summary offline")
    summarize.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Chạy một phase vòng đời C và trả status code."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            output_dir = Path(args.output_dir)
            if output_dir.exists():
                print(f"Existing result at {output_dir}; left untouched")
                return 0
            run_dir, manifest, _ = prepare_prompt_ablation(output_dir, list(args.policies))
            print(f"Run directory: {run_dir}")
            print(f"Scheduled {manifest.expected_trial_count} trials")
            return 0
        if args.command == "run":
            status = _execute_and_finalize(args.run_dir)
            return 0 if status == "complete" else 1
        if args.command == "summarize":
            status = _write_summary_and_report(
                args.run_dir, load_prompt_ablation(args.run_dir)[0]
            )
            return 0 if status in {"complete", "completed_with_errors", "prepared", "incomplete"} else 1
    except KeyboardInterrupt:
        return 130
    except (PromptAblationArtifactError, OSError, ValueError) as exc:
        print(_sanitize_error(exc), file=sys.stderr)
        return 2
    return 2


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
