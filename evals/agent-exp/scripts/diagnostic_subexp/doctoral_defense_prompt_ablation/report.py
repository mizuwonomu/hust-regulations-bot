"""Render Markdown report C từ prompt trace và summary đã lưu."""

from __future__ import annotations

from typing import Any

from diagnostic_subexp.doctoral_defense_prompt_ablation.contracts import VARIANTS


def render_prompt_ablation_report(manifest: Any, traces: list[Any], summary: dict[str, Any]) -> str:
    """Hiển thị composition và từng hướng thay đổi, không che bằng pooled accuracy."""
    lines = ["# Defense few-shot ablation C", "", f"Status: {summary['status']}", "",
             "Baseline: 14 messages, 5 FOLLOW / 1 STOP; ablated: 8 messages, 2 FOLLOW / 1 STOP", "",
             "| Policy | Phase | Q | Arm | Baseline actions | Ablated actions | Valid pairs | Switches |",
             "|---|---|---|---|---|---|---|---|"]
    for row in summary["per_case_arm_effects"]:
        counts = [row["variants"][v]["action_counts"] for v in VARIANTS]
        lines.append(f"| {row['policy']} | {row['phase_id']} | {row['question_id']} | {row['arm_id']} | {counts[0]} | {counts[1]} | {row['complete_valid_pairs']}/{row['scheduled_pairs']} | {row['switch_count']} |")
    lines += ["", "## Coverage", "", "| Policy | Scheduled | Valid | Errors | Missing |", "|---|---|---|---|---|"]
    for policy, row in summary["coverage_by_policy"].items():
        lines.append(f"| {policy} | {row['scheduled']} | {row['valid']} | {row['errors']} | {row['missing']} |")
    lines += ["", "## Repeat completeness", "", "| Policy / phase / arm / variant | Complete / scheduled groups | Comparable / scheduled pairs | Incomplete groups / pairs | Agreement |", "|---|---|---|---|---|"]
    for row in summary["per_case_arm_effects"]:
        for variant, metrics in row["variants"].items():
            coverage = metrics["repeat_coverage"]
            lines.append(f"| {row['policy']} / {row['phase_id']} / {row['arm_id']} / {variant} | {coverage['complete_groups']}/{coverage['scheduled_groups']} | {coverage['comparable_valid_pairs']}/{coverage['scheduled_pairs']} | {coverage['incomplete_groups']} / {coverage['incomplete_pairs']} | {metrics['repeat_pair_agreement']} |")
    lines += ["", "## Batch agreement", "", "| Policy / phase / Q / arm / variant | Complete / scheduled | Incomplete | Agreement |", "|---|---|---|---|"]
    for row in summary["batch_agreement"]:
        lines.append(f"| {row['policy']} / {row['phase_id']} / {row['question_id']} / {row['arm_id']} / {row['variant_id']} | {row['complete_valid_pairs']}/{row['scheduled_pairs']} | {row['incomplete_pairs']} | {row['agreement']} |")
    lines += ["", "## C2 question macro", "", "| Policy | Variant | Metric | Eligible | Defined | Undefined | Mean |", "|---|---|---|---|---|---|---|"]
    for row in summary["c2_macro_accuracy"]:
        for metric in ("selection_accuracy", "decision_accuracy"):
            value = row[metric]
            lines.append(f"| {row['policy']} | {row['variant_id']} | {metric} | {value['eligible_questions']} | {value['defined_questions']} | {value['undefined_questions']} | {value['mean']} |")
    lines += ["", "## A-by-C matched swaps", ""]
    for row in summary["a_by_c_interactions"]:
        lines += [f"### {row['policy']}: {row['arms'][0]} -> {row['arms'][1]}", ""]
        for variant, swap in row["variant_swap_tables"].items():
            lines += [f"{variant}: 3-before-41 -> 41-before-3", f"Complete {swap['complete_valid_pairs']}/{swap['scheduled_pairs']}; incomplete {swap['incomplete_pairs']}; agreement {swap['agreement']}", "", "| From | To | Count |", "|---|---|---|"]
            lines.extend(f"| {cell['from']} | {cell['to']} | {cell['count']} |" for cell in swap["transition_matrix"])
        did = row["decision_accuracy_difference_in_differences"]
        lines += ["", f"Decision-accuracy difference-in-differences: {did['value']}; undefined components: {did['undefined_components']}", f"C effects (ablated - baseline): {did['effects']}", "", "| Orientation | Variant | Numerator | Denominator | Value |", "|---|---|---|---|---|"]
        for orientation, values in did["cells"].items():
            for variant, cell in values.items():
                lines.append(f"| {orientation} | {variant} | {cell['numerator']} | {cell['denominator']} | {cell['value']} |")
    lines += ["", "## Trace evidence", "", "Full paired requests: input_traces.jsonl and prompt_traces.jsonl",
              "Source parity and literal example registry: manifest.json and prompt_traces.jsonl",
              "", "## Detailed measurements", "", "Per-batch accuracy, raw denominators, token usage, repeat consistency, batch agreement, role guards and A-by-C interactions are in summary.json",
              "", summary["interpretation"], ""]
    return "\n".join(lines)


__all__ = ["render_prompt_ablation_report"]
