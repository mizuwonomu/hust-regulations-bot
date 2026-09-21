"""Render Markdown report tái lập được từ artifact của intervention A."""

from __future__ import annotations

from typing import Any

from diagnostic_subexp.candidate_pair_41_3_position.contracts import (
    CandidateInputTrace,
    CandidateManifest,
    CandidateOutputs,
    CandidateSummary,
)
from diagnostic_subexp.shared.contracts import DiagnosticResult


def _ratio_text(ratio: Any) -> str:
    if ratio is None:
        return "n/a"
    if ratio.value is None:
        return f"n/a ({ratio.numerator}/{ratio.denominator})"
    return f"{ratio.value:.4f} ({ratio.numerator}/{ratio.denominator})"


def _schedule_section(manifest: CandidateManifest, traces: dict[str, CandidateInputTrace]) -> str:
    lines = ["### Trial schedule\n"]
    arm_ids = [trial.arm_id for trial in manifest.trials]
    seen: list[str] = []
    for arm_id in arm_ids:
        if arm_id not in seen:
            seen.append(arm_id)
    lines.append("| arm | presented candidates | pair positions | relative order | transform | trace |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for arm_id in seen:
        trace = next((item for item in traces.values() if item.arm_id == arm_id), None)
        if trace is None:
            continue
        lines.append(
            f"| `{arm_id}` | {trace.presented_candidates} | {trace.pair_positions} "
            f"| {trace.relative_pair_order} "
            f"| original "
            f"| `{trace.input_trace_id}` |"
        )
    lines.append("")
    lines.append("| batch | repeat | trial | arm |")
    lines.append("| --- | --- | --- | --- |")
    for trial in manifest.trials:
        lines.append(
            f"| {trial.batch_id} | {trial.repeat_id} | `{trial.trial_id}` | `{trial.arm_id}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _decision_section(
    manifest: CandidateManifest,
    traces: dict[str, CandidateInputTrace],
    results: list[DiagnosticResult],
) -> str:
    lines = ["### Decisions\n"]
    lines.append(
        "| trial | policy | arm | batch | repeat | action | selected article | selected position "
        "| input trace | selection correct | decision correct | outcome | latency ms |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    result_by_key = {(result.policy, result.trial_id): result for result in results}
    for trial in manifest.trials:
        for policy in manifest.policies:
            result = result_by_key.get((policy, trial.trial_id))
            if result is None:
                lines.append(
                    f"| `{trial.trial_id}` | {policy} | `{trial.arm_id}` | {trial.batch_id} "
                    f"| {trial.repeat_id} | missing | | | `{trial.input_trace_id}` | | | | |"
                )
                continue
            action = "error"
            if result.outcome.status == "ok" and result.outcome.decision is not None:
                decision = result.outcome.decision
                action = "STOP" if decision.stop else f"FOLLOW:{decision.dieu}"
            lines.append(
                f"| `{trial.trial_id}` | {policy} | `{trial.arm_id}` | {trial.batch_id} "
                f"| {trial.repeat_id} | {action} | {result.selected_article} "
                f"| {result.selected_position} | `{trial.input_trace_id}` "
                f"| {result.selection_correct} | {result.decision_correct} "
                f"| {result.outcome.status} | {result.outcome.latency_ms:.1f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _metrics_section(summary: CandidateSummary) -> str:
    lines = ["### Metrics\n"]
    lines.append(
        f"- scheduled {summary.scheduled} / valid {summary.valid} / errors {summary.errors} "
        f"/ missing {summary.missing}\n"
        f"- independent_question_count = {summary.independent_question_count}\n"
        "- this is a diagnostic over frozen single-step gate inputs, not a full-loop "
        "retrieval result, not an internal-reasoning trace\n"
    )
    lines.append("**Per-policy coverage and accuracy**\n")
    lines.append("| policy | scheduled | valid | errors | missing | selection accuracy | decision accuracy |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for policy, row in sorted(summary.by_policy.items()):
        lines.append(
            f"| {policy} | {row.scheduled} | {row.valid} | {row.errors} | {row.missing} "
            f"| {_ratio_text(row.selection_accuracy)} | {_ratio_text(row.decision_accuracy)} |"
        )
    lines.append("")
    lines.append("**Action counts per arm, batch and repeat**\n")
    lines.append("| policy | arm | batch | repeat | scheduled | valid | errors | missing | actions |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary.action_counts:
        lines.append(
            f"| {row.policy} | `{row.arm_id}` | {row.batch_id or 'pooled'} "
            f"| {row.repeat_id if row.repeat_id is not None else 'pooled'} "
            f"| {row.scheduled} | {row.valid} | {row.errors} | {row.missing} | {row.action_counts} |"
        )
    lines.append("")
    lines.append("**Accuracy per arm and case**\n")
    lines.append("| policy | arm | case | scheduled | valid | errors | missing | selection | decision |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary.accuracy:
        lines.append(
            f"| {row.policy} | `{row.arm_id}` | `{row.case_id}` | {row.scheduled} | {row.valid} "
            f"| {row.errors} | {row.missing} | {_ratio_text(row.selection_accuracy)} "
            f"| {_ratio_text(row.decision_accuracy)} |"
        )
    lines.append("")
    lines.append("**Within-exact-input repeat consistency**\n")
    lines.append("| policy | arm | batch | scheduled | valid | pairs | matching | consistency |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary.repeat_consistency:
        lines.append(
            f"| {row.policy} | `{row.arm_id}` | {row.batch_id} | {row.scheduled} | {row.valid} "
            f"| {row.valid_pairs} | {row.matching_pairs} | {_ratio_text(row.consistency)} |"
        )
    lines.append("")
    lines.append("**Batch-to-batch agreement**\n")
    lines.append("| policy | arm | repeat | scheduled | valid | agreement | switches |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in summary.batch_agreement:
        lines.append(
            f"| {row.policy} | `{row.arm_id}` | {row.repeat_id} | {row.scheduled} | {row.valid} "
            f"| {_ratio_text(row.agreement)} | {row.switch_count} |"
        )
    lines.append("")
    if summary.candidate_relative_position is not None:
        lines.extend(_candidate_metrics(summary.candidate_relative_position))
    return "\n".join(lines)


def _comparison_table(comparisons: list[Any], title: str) -> list[str]:
    lines = [f"**{title}**\n"]
    lines.append(
        "| policy | comparison | scheduled | valid | incomplete | agreement | switches | transition |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in comparisons:
        lines.append(
            f"| {item.policy} | `{item.comparison_id}` | {item.scheduled} | {item.valid} "
            f"| {item.incomplete} | {_ratio_text(item.agreement)} | {item.switch_count} "
            f"| rows {item.action_keys}: {item.transition} |"
        )
    lines.append("")
    return lines


def _candidate_metrics(outputs: CandidateOutputs) -> list[str]:
    lines = ["**Intervention A slices**\n"]
    lines.append(
        "| policy | arm | article 3 | article 41 | earlier-of-pair | pair selections "
        "| STOP | follows outside the pair | pair-slot positions |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in outputs.pair_selections:
        lines.append(
            f"| {row.policy} | `{row.arm_id}` | {row.article_3_count} | {row.article_41_count} "
            f"| {_ratio_text(row.earlier_of_pair)} | {row.pair_selection_count} | {row.stop_count} "
            f"| {row.outside_pair_follow_count} | {row.absolute_position_counts} |"
        )
    lines.append("")
    lines.extend(_comparison_table(outputs.matched_swaps, "Matched-swap action transitions"))
    return lines


def render_candidate_report(
    manifest: CandidateManifest,
    traces: list[CandidateInputTrace],
    results: list[DiagnosticResult],
    summary: CandidateSummary | None = None,
) -> str:
    """Render report Markdown của A từ evidence đã lưu."""
    trace_by_id = {trace.input_trace_id: trace for trace in traces}
    source_rows = [
        "| experiment definition | embedded in manifest | immutable result evidence |",
        *[
            f"| {label} | `{source.path}` | `{source.sha256}` |"
            for label, source in manifest.sources.items()
        ],
    ]
    lines: list[str] = [
        f"# Fixed diagnostic report: {manifest.run_id}\n",
        f"- suite: `{manifest.suite_id}`\n"
        f"- diagnostic kind: `{manifest.diagnostic_kind}`\n"
        f"- case: `{manifest.case_id}`\n"
        f"- status: `{manifest.status}`\n"
        f"- policies: {manifest.policies}\n"
        f"- execution revision: `{manifest.execution_revision}`\n"
        f"- independent_question_count = {manifest.independent_question_count}\n"
        f"- started at: {manifest.started_at}\n"
        f"- ended at: {manifest.ended_at}\n",
        "## Result-native evidence\n",
        "| role | path | sha256 |",
        "| --- | --- | --- |",
        *source_rows,
        "",
        "## Schedule\n",
        _schedule_section(manifest, trace_by_id),
        "## Decisions\n",
        _decision_section(manifest, trace_by_id, results),
    ]
    if summary is not None:
        lines.append("## Results\n")
        lines.append(_metrics_section(summary))
    else:
        lines.append("## Results\n\nNo results were recorded yet for this prepared run\n")
    lines.append("## Exclusions\n")
    if manifest.exclusions:
        for exclusion in manifest.exclusions:
            lines.append(f"- `{exclusion.case_id}`: {exclusion.reason}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


__all__ = ["render_candidate_report"]
