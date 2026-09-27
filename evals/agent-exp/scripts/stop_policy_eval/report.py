"""Render báo cáo STOP ngắn chỉ từ artifact đã validate của một run"""

from __future__ import annotations

from typing import Any

from stop_policy_eval.contracts import (
    StopInputTrace,
    StopResult,
    StopRunManifest,
    StopSummary,
)

# Giới hạn kết luận bắt buộc phải xuất hiện trong mọi báo cáo
CONCLUSION_LIMITS: tuple[str, ...] = (
    "These figures describe gate decisions on frozen states only; they do not measure answer quality",
    "No retrieval benefit, full-loop cost or end-to-end improvement is established here",
    "Heldout numbers are not evidence of generalization until dev selection is frozen",
    "A stable decision does not isolate reasoning from few-shot or ordering heuristics",
    "Fake-client tests validate harness mechanics, never model judgment",
)


def _ratio_text(ratio: Any) -> str:
    if ratio is None:
        return "n/a"
    if ratio.value is None:
        return f"n/a ({ratio.numerator}/{ratio.denominator})"
    return f"{ratio.value:.4f} ({ratio.numerator}/{ratio.denominator})"


def _mean_text(mean: Any) -> str:
    if mean is None:
        return "n/a"
    if mean.value is None:
        return f"n/a (defined {mean.defined}/{mean.eligible})"
    return f"{mean.value:.4f} (defined {mean.defined}/{mean.eligible})"


def _provenance_section(manifest: StopRunManifest) -> list[str]:
    lines = ["### Provenance\n"]
    lines.append(f"- Run: `{manifest.run_id}` status `{manifest.status}`")
    lines.append(f"- Purpose: `{manifest.run_purpose}`")
    if manifest.run_purpose == "later_hop_capture":
        lines.append(
            "- Source-only decisions: draft-label correctness is not a formal STOP measurement"
        )
    lines.append(
        f"- Variant: `{manifest.variant}` | split `{manifest.split}` | repeats {manifest.repeats} "
        f"| policies {', '.join(manifest.policies)}"
    )
    lines.append(
        f"- Cases source: `{manifest.cases_source.path}` sha256 `{manifest.cases_source.sha256}`"
    )
    lines.append(
        f"- Snapshot source: `{manifest.snapshot_source.path}` "
        f"sha256 `{manifest.snapshot_source.sha256}`"
    )
    lines.append(f"- Corpus sha256: `{manifest.corpus_sha256}`")
    lines.append(
        f"- Execution revision `{manifest.execution_revision}` dirty `{manifest.execution_dirty}` "
        f"| {len(manifest.executed_module_hashes)} executed module hashes"
    )
    if manifest.execution_config is None:
        lines.append(f"- LLM configuration: not recorded ({manifest.execution_config_reason})")
    else:
        config = manifest.execution_config
        lines.append(
            f"- LLM configuration: alias `{config.model_alias}` base `{config.base_url}` "
            f"temperature {config.temperature} max tokens {config.max_completion_tokens} "
            f"thinking {config.enable_thinking} timeout {config.timeout_seconds} "
            f"retries {config.max_retries}"
        )
    lines.append("- Provenance limits: served model weights and store contents are not verified here")
    lines.append("")
    return lines


def _class_section(
    manifest: StopRunManifest,
    traces: list[StopInputTrace],
) -> list[str]:
    lines = ["### Class balance and exclusions\n"]
    for row in manifest.readiness.requirements:
        lines.append(f"- {row.requirement}: {row.eligible_cases} eligible case(s)")
    lines.append(
        f"- Complete for the four required classes: {manifest.readiness.complete} "
        f"(missing: {', '.join(manifest.readiness.missing()) or 'none'})"
    )
    counts: dict[str, int] = {}
    for exclusion in manifest.exclusions:
        counts[exclusion.reason] = counts.get(exclusion.reason, 0) + 1
    rendered = ", ".join(f"{reason} {count}" for reason, count in sorted(counts.items()))
    lines.append(f"- Exclusions: {rendered or 'none'}")
    lines.append(f"- Replay inputs: {len(traces)}")
    lines.append("")
    return lines


def _coverage_section(manifest: StopRunManifest, summary: StopSummary | None) -> list[str]:
    lines = ["### Run completeness\n"]
    if summary is None:
        lines.append(
            f"- Prepared only: {manifest.expected_trial_count} scheduled trial(s), "
            f"{manifest.expected_policy_result_count} result slot(s), no invocation yet"
        )
        lines.append("")
        return lines
    lines.append(
        f"- Totals: scheduled {summary.scheduled} | valid {summary.valid} | "
        f"errors {summary.errors} | missing {summary.missing}"
    )
    for policy, item in sorted(summary.by_policy.items()):
        lines.append(
            f"- {policy}: scheduled {item.scheduled} | valid {item.valid} | errors {item.errors} "
            f"| missing {item.missing}"
        )
    lines.append(
        "- Errors, missing slots and invalid decisions are never counted as STOP"
    )
    lines.append("")
    return lines


def _action_section(summary: StopSummary | None) -> list[str]:
    lines = ["### Action rates by label (valid outputs only)\n"]
    if summary is None:
        lines.append("- No results yet")
        lines.append("")
        return lines
    lines.append(
        "| policy | follow accuracy | false stop rate | stop accuracy | over-hop rate | follow valid | stop valid |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for policy, item in sorted(summary.by_policy.items()):
        rates = item.label_rates
        lines.append(
            f"| {policy} | {_ratio_text(rates.follow_accuracy)} | {_ratio_text(rates.false_stop_rate)} "
            f"| {_ratio_text(rates.stop_accuracy)} | {_ratio_text(rates.over_hop_rate)} "
            f"| {rates.follow_valid} | {rates.stop_valid} |"
        )
    lines.append("")
    lines.append(
        "| policy | action accuracy | selection accuracy | decision accuracy | follow error | "
        "follow missing | stop error | stop missing |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for policy, item in sorted(summary.by_policy.items()):
        lines.append(
            f"| {policy} | {_ratio_text(item.action_accuracy)} | {_ratio_text(item.selection_accuracy)} "
            f"| {_ratio_text(item.decision_accuracy)} | {_ratio_text(item.error_rate['follow'])} "
            f"| {_ratio_text(item.missing_rate['follow'])} | {_ratio_text(item.error_rate['stop'])} "
            f"| {_ratio_text(item.missing_rate['stop'])} |"
        )
    lines.append("")
    lines.append("Scheduled accuracies count every labeled slot; errors and missing slots are incorrect there")
    lines.append("")
    return lines


def _case_macro_section(summary: StopSummary | None) -> list[str]:
    lines = ["### Case-level macro means\n"]
    if summary is None:
        lines.append("- No results yet")
        lines.append("")
        return lines
    lines.append("| policy | cases | follow accuracy | stop accuracy | action | selection | decision |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for policy, item in sorted(summary.by_policy.items()):
        macro = item.case_macro
        lines.append(
            f"| {policy} | {macro.case_count} | {_mean_text(macro.follow_accuracy)} "
            f"| {_mean_text(macro.stop_accuracy)} | {_mean_text(macro.action_accuracy)} "
            f"| {_mean_text(macro.selection_accuracy)} | {_mean_text(macro.decision_accuracy)} |"
        )
    lines.append("")
    lines.append(
        "Undefined case values are excluded and reported through the defined/eligible counts above"
    )
    lines.append("")
    return lines


def _strata_section(summary: StopSummary | None) -> list[str]:
    lines = ["### Strata\n"]
    if summary is None:
        lines.append("- No results yet")
        lines.append("")
        return lines
    for kind in ("semantic_group", "source_hop", "split"):
        rows = [row for row in summary.strata if row.kind == kind]
        if not rows:
            continue
        lines.append(f"- {kind}:")
        for row in rows:
            lines.append(
                f"  - {row.value} [{row.policy}]: scheduled {row.scheduled} | valid {row.valid} "
                f"| errors {row.errors} | missing {row.missing} "
                f"| action {_ratio_text(row.action_accuracy)} "
                f"| selection {_ratio_text(row.selection_accuracy)} "
                f"| decision {_ratio_text(row.decision_accuracy)}"
            )
    lines.append("")
    return lines


def _paired_section(summary: StopSummary | None) -> list[str]:
    lines = ["### Paired comparison\n"]
    if summary is None or summary.paired is None:
        lines.append("- Both first and llm results are required for a paired comparison")
        lines.append("")
        return lines
    paired = summary.paired
    lines.append(
        f"- Policies {', '.join(paired.policies)} | scheduled pairs {paired.scheduled_pairs} "
        f"| valid pairs {paired.valid_pairs} | incomplete pairs {paired.incomplete_pairs}"
    )
    lines.append(f"- Action agreement: {_ratio_text(paired.action_agreement)}")
    lines.append(
        f"- Decision correctness: {paired.policies[0]} wins {paired.a_wins} | "
        f"{paired.policies[1]} wins {paired.b_wins} | ties {paired.ties}"
    )
    lines.append(f"- Trial ids with different actions: {len(paired.different_trial_ids)}")
    lines.append("")
    return lines


def _tradeoff_section(summary: StopSummary | None) -> list[str]:
    lines = ["### STOP/FOLLOW trade-off\n"]
    if summary is None:
        lines.append("- No results yet")
        lines.append("")
        return lines
    for policy, item in sorted(summary.by_policy.items()):
        rates = item.label_rates
        lines.append(
            f"- {policy}: over-hop rate {_ratio_text(rates.over_hop_rate)} against false stop rate "
            f"{_ratio_text(rates.false_stop_rate)}; report both sides together"
        )
    lines.append("")
    return lines


def _limits_section() -> list[str]:
    lines = ["### Conclusion limits\n"]
    lines.extend(f"- {limit}" for limit in CONCLUSION_LIMITS)
    lines.append("")
    return lines


def render_stop_report(
    manifest: StopRunManifest,
    traces: list[StopInputTrace],
    results: list[StopResult],
    summary: StopSummary | None,
) -> str:
    """Render báo cáo của một run, chỉ dùng field có thật trong artifact

    Params:
    - manifest: manifest đã validate của run
    - traces: input trace đã lưu
    - results: result compact đã reload, rỗng khi run mới prepared
    - summary: summary đã tính, None khi run chưa có kết quả
    """
    lines = [
        "# STOP policy run report",
        "",
        f"- Experiment: {manifest.experiment}",
        f"- Result records: {len(results)}",
        "",
    ]
    lines.extend(_provenance_section(manifest))
    lines.extend(_class_section(manifest, traces))
    lines.extend(_coverage_section(manifest, summary))
    lines.extend(_action_section(summary))
    lines.extend(_case_macro_section(summary))
    lines.extend(_strata_section(summary))
    lines.extend(_paired_section(summary))
    lines.extend(_tradeoff_section(summary))
    lines.extend(_limits_section())
    return "\n".join(lines).rstrip("\n") + "\n"


def render_source_capture_report(
    manifest: StopRunManifest,
    results: list[StopResult],
) -> str:
    """Render quyết định source run, không tổng hợp action correctness"""
    lines = [
        "# Hop-0 source decisions",
        "",
        f"- Run: {manifest.run_id} status {manifest.status}",
        f"- Purpose: {manifest.run_purpose}",
        f"- Policy: {', '.join(manifest.policies)} | repeats: {manifest.repeats}",
        f"- Scheduled source trials: {manifest.expected_trial_count}",
        "- Source-only run: decisions are used to capture one-edge successor states",
        "- Draft-label correctness fields are not an evaluation result",
        "",
        "## Decisions",
        "",
    ]
    for result in results:
        decision = result.outcome.decision
        if result.outcome.status != "ok" or decision is None:
            detail = result.outcome.error.category if result.outcome.error else "invalid"
            rendered = f"error ({detail})"
        elif decision.stop:
            rendered = "STOP"
        else:
            rendered = f"FOLLOW Điều {decision.dieu}"
        lines.append(f"- {result.trial_id}: {rendered}")
    return "\n".join(lines).rstrip("\n") + "\n"


__all__ = ["CONCLUSION_LIMITS", "render_source_capture_report", "render_stop_report"]
