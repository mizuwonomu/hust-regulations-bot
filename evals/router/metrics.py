"""Chấm điểm offline artifacts của experiment router và ghi summary.json.

Không gọi API nào: chỉ đọc manifest.json + trials.jsonl, tính accuracy, recall, confusion,
repeat consistency, latency, usage và cost theo đúng denominator của Metrics Contract
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from evals.router.adapters import ArmName
from evals.router.corpus import ROUTE_LABELS, RouteLabel, corpus_sha256, load_corpus

SUMMARY_NAME = "summary.json"
TRIAL_KEY_FIELDS = ("case_index", "arm", "repeat")


class ScoringError(RuntimeError):
    """Artifact không chấm được: thiếu file, JSON hỏng, trial trùng key hoặc sai identity"""


@dataclass(frozen=True)
class RouterSummary:
    """Kết quả chấm một run: per-case outcomes, hai arm, và các trial lệch/lỗi/thiếu"""

    run: dict[str, Any]
    cases: list[dict[str, Any]]
    arms: dict[str, dict[str, Any]]
    disagreements: list[dict[str, Any]]
    format_invalid: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    missing: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Payload JSON của summary.json"""
        return {
            "run": self.run,
            "cases": self.cases,
            "arms": self.arms,
            "disagreements": self.disagreements,
            "format_invalid": self.format_invalid,
            "errors": self.errors,
            "missing": self.missing,
        }


def read_trials(trials_path: Path | str) -> list[dict[str, Any]]:
    """Đọc trials.jsonl, fail rõ ràng thay vì bỏ qua dòng hỏng

    trials_path: file JSONL do runner ghi
    """
    path = Path(trials_path)
    if not path.exists():
        raise ScoringError(f"thiếu file trials: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ScoringError(f"{path}:{line_number} không parse được ({error})") from error
        if not isinstance(row, dict):
            raise ScoringError(f"{path}:{line_number} không phải JSON object")
        rows.append(row)
    return rows


def score_run(manifest_path: Path | str, trials_path: Path | str) -> RouterSummary:
    """Chấm một run đã lưu mà không gọi thêm API nào

    manifest_path: manifest.json của run, nguồn của corpus/gold count/pricing/repeats
    trials_path: trials.jsonl của cùng run
    """
    manifest = _read_manifest(manifest_path)
    rows = read_trials(trials_path)
    cases_count = int(manifest["corpus"]["rows"])
    repeats = int(manifest["schedule"]["repeats"])
    pricing: Mapping[str, Mapping[str, Any]] = manifest.get("pricing", {})
    arms = list(manifest["schedule"]["arms"])
    reference = _load_corpus_reference(manifest)
    _check_rows(
        rows,
        cases_count=cases_count,
        repeats=repeats,
        arms=arms,
        reference=reference,
    )

    summaries: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        summaries[arm] = _score_arm(
            arm,
            arm_rows,
            cases_count=cases_count,
            repeats=repeats,
            gold_counts=manifest["corpus"]["gold_counts"],
            pricing=pricing.get(arm, {}),
        )

    decisions = [row for row in rows if row["executed_branch"] in ROUTE_LABELS]
    summary = RouterSummary(
        run={
            "manifest": str(Path(manifest_path)),
            "trials": str(Path(trials_path)),
            "created_at": manifest.get("created_at"),
            "corpus": manifest["corpus"],
            "request_fingerprint": manifest.get("request_fingerprint"),
            "repeats": repeats,
            "scheduled_calls": cases_count * repeats * len(arms),
            "recorded_trials": len(rows),
            "scored_decisions": len(decisions),
            "scope_note": (
                "30 case x repeats là exploratory: không đủ để kết luận robustness hay "
                "production readiness; exact-repeat consistency không gộp probe paraphrase "
                "hay thứ tự choice"
            ),
            "models": {arm: summaries[arm]["model_ids"] for arm in arms},
        },
        arms=summaries,
        cases=_case_outcomes(
            rows,
            cases_count=cases_count,
            repeats=repeats,
            arms=arms,
            reference=reference,
        ),
        disagreements=[
            _review_row(row, reference)
            for row in decisions
            if row["executed_branch"] != row["gold"]
        ],
        format_invalid=[
            _review_row(row, reference) for row in rows if row["format_valid"] is False
        ],
        errors=[_review_row(row, reference) for row in rows if row["error_type"]],
        missing=_missing_rows(
            rows,
            cases_count=cases_count,
            repeats=repeats,
            arms=arms,
            reference=reference,
        ),
    )
    return summary


def write_summary(summary: RouterSummary, manifest_path: Path | str) -> Path:
    """Ghi summary.json độc quyền cạnh manifest, không bao giờ ghi đè báo cáo cũ

    summary: kết quả chấm
    manifest_path: manifest của run, dùng để chọn thư mục đích
    """
    target = Path(manifest_path).parent / SUMMARY_NAME
    if target.exists():
        raise ScoringError(f"{target} đã tồn tại, không ghi đè báo cáo cũ")
    with target.open("x", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point CLI: chấm run đã lưu và ghi summary.json

    argv: tham số dòng lệnh, None thì lấy từ sys.argv
    """
    parser = argparse.ArgumentParser(
        prog="python -m evals.router.metrics",
        description="Chấm offline manifest.json + trials.jsonl của experiment router",
    )
    parser.add_argument("--manifest", required=True, help="manifest.json của run")
    parser.add_argument("--trials", required=True, help="trials.jsonl của run")
    args = parser.parse_args(argv)
    try:
        summary = score_run(args.manifest, args.trials)
        target = write_summary(summary, args.manifest)
    except ScoringError as error:
        print(f"lỗi: {error}", file=sys.stderr)
        return 1
    print(render_report(summary))
    print(f"Đã ghi {target}")
    return 0


def render_report(summary: RouterSummary) -> str:
    """Báo cáo ngắn cho người đọc, đủ số liệu chính của từng arm

    summary: kết quả chấm
    """
    lines = [
        f"Run {summary.run['trials']} | {summary.run['recorded_trials']}/{summary.run['scheduled_calls']} trial",
    ]
    for arm, stats in summary.arms.items():
        lines.append(
            f"- {arm}: scheduled accuracy {_fmt(stats['scheduled_accuracy'])} | "
            f"decision accuracy {_fmt(stats['decision_accuracy'])} | "
            f"recall scheduled/decided RAG {_fmt(stats['recall_scheduled']['RAG'])}"
            f"/{_fmt(stats['recall_decided']['RAG'])} chat {_fmt(stats['recall_scheduled']['chat'])}"
            f"/{_fmt(stats['recall_decided']['chat'])} | "
            f"RAG->chat {stats['rag_to_chat']} | "
            f"error {stats['errors']} missing {stats['missing']} | "
            f"format valid {stats['format_valid']['valid']}/invalid {stats['format_valid']['invalid']} | "
            f"consistency {_fmt(stats['consistency_mean'])} "
            f"({stats['consistency_eligible']}/{stats['consistency_total']} case đủ 3 nhánh) | "
            f"p50 {_fmt(stats['latency_ms']['p50'])}ms p95 {_fmt(stats['latency_ms']['p95'])}ms | "
            f"cost {_fmt(stats['cost_usd'])} USD"
        )
    lines.append(
        f"Disagreement {len(summary.disagreements)} | format invalid {len(summary.format_invalid)} | "
        f"error {len(summary.errors)} | missing {len(summary.missing)}"
    )
    return "\n".join(lines)


def _read_manifest(manifest_path: Path | str) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        raise ScoringError(f"thiếu file manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ScoringError(f"{path} không parse được ({error})") from error
    for field in ("corpus", "schedule", "request_fingerprint"):
        if field not in manifest:
            raise ScoringError(f"{path} thiếu field {field}")
    return manifest


def _check_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cases_count: int,
    repeats: int,
    arms: Sequence[str],
    reference: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    seen: set[tuple[int, str, int]] = set()
    for row in rows:
        # Các field dưới đây đều được scorer truy cập trực tiếp, thiếu là artifact hỏng
        for field in (
            *TRIAL_KEY_FIELDS,
            "gold",
            "executed_branch",
            "format_valid",
            "error_type",
            "latency_ms",
            "usage",
        ):
            if field not in row:
                raise ScoringError(f"trial thiếu field {field}: {row}")
        if not isinstance(row["arm"], str):
            raise ScoringError(f"trial có arm không phải string: {row['arm']!r}")
        for field in ("case_index", "repeat"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ScoringError(f"trial có {field} không phải số nguyên: {value!r}")
        if row["gold"] not in ROUTE_LABELS:
            raise ScoringError(f"trial có gold lạ {row['gold']!r}")
        if row["executed_branch"] is not None and row["executed_branch"] not in ROUTE_LABELS:
            raise ScoringError(
                f"trial có executed_branch lạ: {row['executed_branch']!r}"
            )
        if row["format_valid"] is not None and not isinstance(row["format_valid"], bool):
            raise ScoringError(f"trial có format_valid không phải bool: {row['format_valid']!r}")
        if row["error_type"] is not None and not isinstance(row["error_type"], str):
            raise ScoringError(f"trial có error_type không phải string: {row['error_type']!r}")
        if row["error_type"] is not None and row["executed_branch"] is not None:
            # Lỗi không bao giờ được biến thành route, nên row vừa lỗi vừa có nhánh là mâu thuẫn
            raise ScoringError(
                f"trial vừa có error_type {row['error_type']!r} vừa có executed_branch"
            )
        _check_usage_shape(row)
        if isinstance(row["latency_ms"], bool) or not isinstance(row["latency_ms"], (int, float)):
            raise ScoringError(f"trial có latency_ms không phải số: {row['latency_ms']!r}")
        key = (row["case_index"], row["arm"], row["repeat"])
        if row["arm"] not in arms:
            raise ScoringError(
                f"trial có arm {row['arm']!r} không thuộc manifest {list(arms)}"
            )
        if not 0 <= row["case_index"] < cases_count:
            raise ScoringError(f"trial có case index ngoài phạm vi: {row['case_index']}")
        if not 0 <= row["repeat"] < repeats:
            raise ScoringError(f"trial có repeat ngoài phạm vi: {row['repeat']}")
        if reference is not None:
            expected_gold = reference[row["case_index"]]["type"]
            if row["gold"] != expected_gold:
                raise ScoringError(
                    f"trial {key} có gold {row['gold']!r} khác corpus đã hash {expected_gold!r}"
                )
        if key in seen:
            raise ScoringError(f"trial trùng key {key}")
        seen.add(key)


def _check_usage_shape(row: Mapping[str, Any]) -> None:
    usage = row["usage"]
    if usage is None:
        return
    if not isinstance(usage, dict):
        raise ScoringError(f"trial có usage không phải object hoặc null: {usage!r}")
    for field in ("input_tokens", "output_tokens"):
        if field not in usage:
            raise ScoringError(f"trial usage thiếu field {field}: {usage}")
        value = usage[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ScoringError(f"trial usage có {field} không phải số nguyên: {value!r}")


def _score_arm(
    arm: ArmName,
    rows: Sequence[Mapping[str, Any]],
    *,
    cases_count: int,
    repeats: int,
    gold_counts: Mapping[str, int],
    pricing: Mapping[str, Any],
) -> dict[str, Any]:
    scheduled = cases_count * repeats
    missing = max(scheduled - len(rows), 0)
    scheduled_by_gold = {
        label: gold_counts.get(label, 0) * repeats for label in ROUTE_LABELS
    }
    decisions = [row for row in rows if row["executed_branch"] in ROUTE_LABELS]
    error_rows = [row for row in rows if row["error_type"]]
    unexpected = [
        row
        for row in rows
        if row["executed_branch"] not in ROUTE_LABELS and not row["error_type"]
    ]
    correct = [row for row in decisions if row["executed_branch"] == row["gold"]]
    confusion = {
        gold: {executed: 0 for executed in ROUTE_LABELS} for gold in ROUTE_LABELS
    }
    for row in decisions:
        confusion[row["gold"]][row["executed_branch"]] += 1
    correct_by_gold = {
        label: sum(1 for row in correct if row["gold"] == label) for label in ROUTE_LABELS
    }
    decided_by_gold = {
        label: sum(1 for row in decisions if row["gold"] == label) for label in ROUTE_LABELS
    }
    consistency_mean, consistency_by_gold, consistency_eligible = _consistency(
        rows,
        cases_count=cases_count,
        repeats=repeats,
    )
    latency = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    input_tokens = sum(
        int((row.get("usage") or {}).get("input_tokens") or 0) for row in rows
    )
    output_tokens = sum(
        int((row.get("usage") or {}).get("output_tokens") or 0) for row in rows
    )
    missing_usage = sum(
        1
        for row in rows
        if not row.get("usage")
        or row["usage"].get("input_tokens") is None
        or row["usage"].get("output_tokens") is None
    )
    cost_usd, cost_blockers = _cost_usd(
        rows,
        pricing=pricing,
        missing_usage=missing_usage,
        missing_trials=missing,
    )
    return {
        "arm": arm,
        "scheduled": scheduled,
        "recorded": len(rows),
        "missing": missing,
        "errors": len(error_rows),
        "unexpected": len(unexpected),
        "decisions": len(decisions),
        "correct": len(correct),
        "scheduled_accuracy": len(correct) / scheduled if scheduled else None,
        "decision_accuracy": len(correct) / len(decisions) if decisions else None,
        # recall_scheduled tính cả error và trial thiếu là miss, recall_decided chỉ tính trên decision
        "recall_scheduled": {
            label: correct_by_gold[label] / scheduled_by_gold[label]
            if scheduled_by_gold[label]
            else None
            for label in ROUTE_LABELS
        },
        "recall_decided": {
            label: correct_by_gold[label] / decided_by_gold[label]
            if decided_by_gold[label]
            else None
            for label in ROUTE_LABELS
        },
        "recall_scheduled_by_gold": scheduled_by_gold,
        "recall_decided_by_gold": decided_by_gold,
        "confusion": confusion,
        "rag_to_chat": confusion["RAG"]["chat"],
        "rag_to_chat_rate": confusion["RAG"]["chat"] / scheduled_by_gold["RAG"]
        if scheduled_by_gold["RAG"]
        else None,
        "format_valid": _format_counts(rows),
        "consistency_mean": consistency_mean,
        "consistency_by_gold": consistency_by_gold,
        "consistency_eligible": consistency_eligible,
        "consistency_total": cases_count,
        "latency_ms": {
            "p50": _percentile(latency, 0.50),
            "p95": _percentile(latency, 0.95),
            "max": max(latency) if latency else None,
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "missing_usage": missing_usage,
        },
        "cost_usd": cost_usd,
        "cost_complete": cost_usd is not None,
        "cost_blockers": cost_blockers,
        "model_ids": sorted({row["model"] for row in rows if row.get("model")}),
    }


def _pair_consistency(branches: Sequence[str | None]) -> float | None:
    """Tỉ lệ cặp nhánh khớp của một query; None nếu thiếu nhánh hoặc có nhánh không hợp lệ"""
    if len(branches) < 2 or any(branch not in ROUTE_LABELS for branch in branches):
        return None
    pairs = list(combinations(branches, 2))
    return sum(1 for left, right in pairs if left == right) / len(pairs)


def _case_outcomes(
    rows: Sequence[Mapping[str, Any]],
    *,
    cases_count: int,
    repeats: int,
    arms: Sequence[str],
    reference: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Outcome từng case cho toàn bộ corpus: nhánh theo repeat, đúng/sai, lỗi, thiếu, consistency"""
    by_key = {(row["case_index"], row["arm"], row["repeat"]): row for row in rows}
    cases: list[dict[str, Any]] = []
    for case_index in range(cases_count):
        annotation = (
            reference[case_index] if reference and case_index < len(reference) else {}
        )
        recorded = sorted(
            (row for row in rows if row["case_index"] == case_index),
            key=lambda row: (row["arm"], row["repeat"]),
        )
        gold = annotation.get("type") or (recorded[0]["gold"] if recorded else None)
        per_arm: dict[str, dict[str, Any]] = {}
        for arm in arms:
            branches: list[str | None] = []
            errors = 0
            missing = 0
            for repeat_index in range(repeats):
                row = by_key.get((case_index, arm, repeat_index))
                if row is None:
                    missing += 1
                    continue
                branches.append(row["executed_branch"])
                if row["error_type"]:
                    errors += 1
            decisions = [branch for branch in branches if branch in ROUTE_LABELS]
            per_arm[arm] = {
                "branches": branches,
                "decisions": len(decisions),
                "correct": sum(1 for branch in decisions if branch == gold),
                "errors": errors,
                "missing": missing,
                "consistency": _pair_consistency(branches),
            }
        cases.append(
            {
                "case_index": case_index,
                "id": annotation.get("id", recorded[0].get("id") if recorded else None),
                "group": annotation.get(
                    "group", recorded[0].get("group") if recorded else None
                ),
                "gold": gold,
                "arms": per_arm,
            }
        )
    return cases


def _consistency(
    rows: Sequence[Mapping[str, Any]],
    *,
    cases_count: int,
    repeats: int,
) -> tuple[float | None, dict[str, float | None], int]:
    """Repeat consistency: trung bình tỉ lệ cặp nhánh khớp của từng query đủ 3 nhánh hợp lệ"""
    by_case: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(int(row["case_index"]), []).append(row)
    per_case: list[float | None] = []
    gold_per_case: list[RouteLabel | None] = []
    for case_index in range(cases_count):
        case_rows = sorted(by_case.get(case_index, []), key=lambda row: row["repeat"])
        gold = case_rows[0]["gold"] if case_rows else None
        gold_per_case.append(gold if gold in ROUTE_LABELS else None)
        if len(case_rows) != repeats:
            per_case.append(None)
            continue
        per_case.append(_pair_consistency([row["executed_branch"] for row in case_rows]))
    scores = [score for score in per_case if score is not None]
    by_gold = {
        label: fmean(
            [
                score
                for score, gold in zip(per_case, gold_per_case)
                if score is not None and gold == label
            ]
        )
        if any(
            score is not None and gold == label for score, gold in zip(per_case, gold_per_case)
        )
        else None
        for label in ROUTE_LABELS
    }
    return (fmean(scores) if scores else None), by_gold, len(scores)


def _format_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "valid": sum(1 for row in rows if row["format_valid"] is True),
        "invalid": sum(1 for row in rows if row["format_valid"] is False),
        "unknown": sum(1 for row in rows if row["format_valid"] is None),
    }


def _cost_usd(
    rows: Sequence[Mapping[str, Any]],
    *,
    pricing: Mapping[str, Any],
    missing_usage: int,
    missing_trials: int,
) -> tuple[float | None, list[str]]:
    """Tính cost chỉ từ usage đã ghi và pricing snapshot; thiếu cơ sở thì trả None

    rows: trial đã ghi của arm
    pricing: entry giá của arm
    missing_usage: số attempt không có usage đầy đủ
    missing_trials: số attempt đã lập lịch nhưng không có trong artifact
    """
    blockers: list[str] = []
    rate_in = pricing.get("usd_per_million_input")
    rate_out = pricing.get("usd_per_million_output")
    priced_model = pricing.get("model_id")
    if rate_in is None or rate_out is None:
        blockers.append("missing_pricing")
    # Trial thiếu cũng là usage không biết, nên cost của cả arm là không đầy đủ
    if missing_trials:
        blockers.append("missing_trials")
    if missing_usage:
        blockers.append("missing_usage")
    # Giá chỉ áp dụng cho model đã niêm yết, nên model phục vụ khác là không có cơ sở tính
    served_models = sorted({row["model"] for row in rows if row.get("model")})
    if priced_model:
        if not served_models and rows:
            blockers.append("served_model_unknown")
        if any(model != priced_model for model in served_models):
            blockers.append("served_model_not_priced")
    if blockers:
        return None, blockers
    total = 0.0
    for row in rows:
        usage = row["usage"]
        total += (usage["input_tokens"] / 1_000_000) * rate_in
        total += (usage["output_tokens"] / 1_000_000) * rate_out
    return total, blockers


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _review_row(row: Mapping[str, Any], reference: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    case_index = int(row["case_index"])
    annotation = reference[case_index] if reference and case_index < len(reference) else {}
    return {
        "case_index": case_index,
        "id": row.get("id", annotation.get("id")),
        "group": row.get("group", annotation.get("group")),
        "arm": row["arm"],
        "repeat": row["repeat"],
        "gold": row["gold"],
        "executed_branch": row["executed_branch"],
        "format_valid": row.get("format_valid"),
        "raw_answer": row.get("raw_answer"),
        "error_type": row.get("error_type"),
    }


def _missing_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cases_count: int,
    repeats: int,
    arms: Sequence[str],
    reference: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    recorded = {(row["case_index"], row["arm"], row["repeat"]) for row in rows}
    missing: list[dict[str, Any]] = []
    for case_index in range(cases_count):
        for arm in arms:
            for repeat_index in range(repeats):
                if (case_index, arm, repeat_index) in recorded:
                    continue
                annotation = (
                    reference[case_index]
                    if reference and case_index < len(reference)
                    else {}
                )
                missing.append(
                    {
                        "case_index": case_index,
                        "id": annotation.get("id"),
                        "group": annotation.get("group"),
                        "arm": arm,
                        "repeat": repeat_index,
                    }
                )
    return missing


def _load_corpus_reference(manifest: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Đọc lại corpus đã dùng (id/group/gold type) nếu file còn và hash còn khớp"""
    corpus = manifest.get("corpus") or {}
    path = corpus.get("path")
    if not path:
        return None
    try:
        if corpus_sha256(path) != corpus.get("sha256"):
            return None
        return [
            {"id": case.id, "group": case.group, "type": case.type}
            for case in load_corpus(path)
        ]
    except Exception:  # noqa: BLE001 - thiếu reference không được làm hỏng việc chấm
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
