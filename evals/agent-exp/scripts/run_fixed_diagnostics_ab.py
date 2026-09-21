"""Entrypoint prepare, run và summarize cho hai suite diagnostic A và B.

Orchestrator chỉ gọi API công khai của hai harness con; mỗi harness giữ loader,
metric, report và cấu trúc manifest riêng.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    candidate for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (candidate / "pyproject.toml").is_file()
)
if __package__ in {None, ""}:
    for bootstrap_path in (REPO_ROOT, SCRIPT_DIR):
        value = str(bootstrap_path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

from diagnostic_subexp.article_42_observation_block_order import cli as observation_cli
from diagnostic_subexp.candidate_pair_41_3_position import cli as candidate_cli
from run_experiments import _sanitize_error

HARNESSES = {
    "candidate-pair-41-3-position": candidate_cli,
    "article-42-observation-block-order": observation_cli,
}
PHASE_SUCCESS: dict[str, frozenset[str]] = {
    "prepare": frozenset({"prepared", "existing-result"}),
    "run": frozenset({"complete"}),
    "summarize": frozenset({"complete", "completed_with_errors", "prepared", "incomplete"}),
}


@dataclass(frozen=True, slots=True)
class SuiteOutcome:
    """Kết quả một phase của một suite con độc lập."""

    name: str
    status: str
    detail: str


def _suite_paths(output_root: Path) -> dict[str, Path]:
    """Ánh xạ tên suite sang thư mục con tất định của output root."""
    return {name: Path(output_root) / name for name in HARNESSES}


def _prepare_suites(output_root: Path, policies: list[str]) -> list[SuiteOutcome]:
    """Chỉ kiểm tra sự tồn tại của output root, rồi materialize cả hai suite nếu trống."""
    output_root = Path(output_root)
    if output_root.exists():
        # Requested output root đã tồn tại là boundary chỉ đọc; không mở file con nào
        return [
            SuiteOutcome(
                name=name,
                status="existing-result",
                detail=f"{output_root / name}: existing result left untouched",
            )
            for name in HARNESSES
        ]
    child_paths = _suite_paths(output_root)
    validated = {
        name: module.validate_run(None, list(policies)) for name, module in HARNESSES.items()
    }
    created: list[Path] = []
    outcomes: list[SuiteOutcome] = []
    try:
        for name, module in HARNESSES.items():
            child = child_paths[name]
            if name not in validated:
                outcomes.append(
                    SuiteOutcome(
                        name=name,
                        status="existing-result",
                        detail=f"{child}: existing result left untouched",
                    )
                )
                continue
            outcome = module.materialize_run(validated[name], child, list(policies))
            if outcome.status == "existing-result":
                outcomes.append(
                    SuiteOutcome(
                        name=name,
                        status="existing-result",
                        detail=f"{child}: existing result left untouched",
                    )
                )
                continue
            created.append(outcome.run_dir)
            outcomes.append(
                SuiteOutcome(
                    name=name,
                    status=outcome.manifest.status,
                    detail=(
                        f"{outcome.run_dir}: {outcome.plan.expected_trial_count} trials, "
                        f"{outcome.plan.expected_policy_result_count} result slots, "
                        f"status {outcome.manifest.status}"
                    ),
                )
            )
    except BaseException:
        # Không để lại run con dở dang khi một suite materialize thất bại
        for run_dir in created:
            shutil.rmtree(run_dir, ignore_errors=True)
        output_root = Path(output_root)
        if output_root.exists() and not any(output_root.iterdir()):
            output_root.rmdir()
        raise
    return outcomes


def _run_suites(output_root: Path) -> list[SuiteOutcome]:
    """Chạy hai suite đã persist độc lập và giữ trạng thái riêng cho từng suite."""
    outcomes: list[SuiteOutcome] = []
    for name, module in HARNESSES.items():
        run_dir = _suite_paths(output_root)[name]
        if not (run_dir / "manifest.json").exists():
            outcomes.append(
                SuiteOutcome(name=name, status="missing", detail=f"{run_dir}: no prepared run")
            )
            continue
        interrupted = False
        failure: Exception | None = None
        try:
            module.execute_run(run_dir)
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exc:  # noqa: BLE001
            failure = exc
        status = module.finalize_run(run_dir)
        if interrupted:
            raise KeyboardInterrupt
        if failure is not None or status is None:
            detail = f"{run_dir}: {_sanitize_error(failure) if failure else 'could not finalize'}"
            print(f"{name} failed: {detail}", file=sys.stderr)
            outcomes.append(SuiteOutcome(name=name, status="failed", detail=detail))
            continue
        outcomes.append(SuiteOutcome(name=name, status=status, detail=f"{run_dir}: status {status}"))
    return outcomes


def _summarize_suites(output_root: Path) -> list[SuiteOutcome]:
    """Tính lại summary và report của từng suite hoàn toàn offline."""
    outcomes: list[SuiteOutcome] = []
    for name, module in HARNESSES.items():
        run_dir = _suite_paths(output_root)[name]
        if not (run_dir / "manifest.json").exists():
            outcomes.append(
                SuiteOutcome(name=name, status="missing", detail=f"{run_dir}: no prepared run")
            )
            continue
        try:
            status = module.summarize_run(run_dir)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            detail = f"{run_dir}: {_sanitize_error(exc)}"
            print(f"{name} failed: {detail}", file=sys.stderr)
            outcomes.append(SuiteOutcome(name=name, status="failed", detail=detail))
            continue
        outcomes.append(
            SuiteOutcome(
                name=name,
                status=status,
                detail=f"{run_dir}: summary {run_dir / 'summary.json'}",
            )
        )
    return outcomes


def _report_outcomes(phase: str, outcomes: list[SuiteOutcome]) -> int:
    """In trạng thái từng suite và trả exit code tổng hợp của phase."""
    for outcome in outcomes:
        print(f"[{phase}] {outcome.name}: {outcome.status} - {outcome.detail}")
    failed = [outcome for outcome in outcomes if outcome.status not in PHASE_SUCCESS[phase]]
    return 0 if not failed else 1


def _prepare_command(args: argparse.Namespace) -> int:
    """Prepare cả hai suite vào hai thư mục con riêng biệt."""
    return _report_outcomes("prepare", _prepare_suites(Path(args.output_root), list(args.policies)))


def _run_command(args: argparse.Namespace) -> int:
    """Chạy cả hai lịch đã lưu và báo trạng thái riêng cho từng suite."""
    return _report_outcomes("run", _run_suites(Path(args.output_root)))


def _summarize_command(args: argparse.Namespace) -> int:
    """Tính lại summary và report của cả hai suite."""
    return _report_outcomes("summarize", _summarize_suites(Path(args.output_root)))


def _build_parser() -> argparse.ArgumentParser:
    """Tạo parser ba phase cho runner dual-suite."""
    parser = argparse.ArgumentParser(description="Fixed-internal diagnostic A+B runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="validate both embedded definitions and materialize two unstarted runs",
    )
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--policies", nargs="+", choices=["first", "llm"], default=["first", "llm"])

    run = subparsers.add_parser("run", help="execute both prepared diagnostic schedules")
    run.add_argument("--output-root", type=Path, required=True)

    summarize = subparsers.add_parser(
        "summarize",
        help="recompute both summaries and reports offline",
    )
    summarize.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Điều phối một phase dual-suite và trả exit code đã quy ước."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return _prepare_command(args)
        if args.command == "run":
            return _run_command(args)
        if args.command == "summarize":
            return _summarize_command(args)
    except KeyboardInterrupt:
        print("Run interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


__all__ = ["HARNESSES", "SuiteOutcome", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
