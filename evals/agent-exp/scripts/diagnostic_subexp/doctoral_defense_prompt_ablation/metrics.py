"""Tổng hợp C theo phase, từng case và cặp treatment với mẫu số tách biệt."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from contracts import DecisionOutcome
from metrics import action_key_from_outcome

from diagnostic_subexp.doctoral_defense_prompt_ablation.contracts import PHASES, VARIANTS
from diagnostic_subexp.shared.contracts import DiagnosticResult


def action_label(outcome: DecisionOutcome | None) -> str | None:
    """Đổi outcome hợp lệ thành nhãn STOP hoặc FOLLOW:<id>."""
    if outcome is None or outcome.status != "ok":
        return None
    key = action_key_from_outcome(outcome)
    if key is None:
        return None
    if key.action == "stop":
        return "STOP"
    return f"FOLLOW:{key.dieu}"


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    """Giữ numerator/denominator cùng giá trị, undefined khi mẫu số bằng không."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _run_status(manifest: Any, results: list[DiagnosticResult]) -> str:
    """Suy ra trạng thái C mà không phụ thuộc module artifact."""
    scheduled = {
        (policy, trial.trial_id) for policy in manifest.policies for trial in manifest.trials
    }
    completed = {(result.policy, result.trial_id) for result in results}
    if completed != scheduled:
        return "incomplete"
    if any(result.outcome.status == "error" for result in results):
        return "completed_with_errors"
    return "complete"


def summarize_prompt_ablation(
    manifest: Any,
    cases: list[Any],
    input_traces: list[Any],
    traces: list[Any],
    results: list[DiagnosticResult],
) -> dict[str, Any]:
    """Tính coverage, accuracy và paired effects hoàn toàn từ artifact đã verify."""
    case_by_id = {case.case_id: case for case in cases}
    trace_by_id = {trace.prompt_trace_id: trace for trace in traces}
    result_by_id = {(result.policy, result.trial_id): result for result in results}

    def measure(policy: str, trials: list[Any]) -> dict[str, Any]:
        records = [result_by_id.get((policy, trial.trial_id)) for trial in trials]
        valid = [record for record in records if record is not None and record.outcome.status == "ok"]
        errors = sum(record is not None and record.outcome.status == "error" for record in records)
        actions = Counter(action_label(record.outcome) for record in valid)
        follows = [record for record in valid if not record.outcome.decision.stop]
        selection_slots = sum(case_by_id[trial.case_id].expected_action == "follow" for trial in trials)
        usage = [
            record.outcome.usage
            for record in records
            if record is not None and record.outcome.usage is not None
        ]
        repeat_groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for trial, record in zip(trials, records):
            repeat_groups[(trial.batch_id, trial.prompt_trace_id)].append(record)
        valid_groups = [
            [
                action_label(record.outcome)
                for record in group
                if record is not None and record.outcome.status == "ok"
            ]
            for group in repeat_groups.values()
        ]
        comparisons = [
            left == right for labels in valid_groups for left, right in combinations(labels, 2)
        ]
        scheduled_pairs = sum(
            len(group) * (len(group) - 1) // 2 for group in repeat_groups.values()
        )
        complete_groups = sum(
            len(valid_group) == len(group)
            for valid_group, group in zip(valid_groups, repeat_groups.values())
        )
        return {
            "scheduled": len(trials),
            "valid": len(valid),
            "errors": errors,
            "missing": len(trials) - len(valid) - errors,
            "action_counts": dict(sorted(actions.items())),
            "stop_rate": ratio(actions["STOP"], len(valid)),
            "follow_rate": ratio(len(follows), len(valid)),
            "selection_accuracy": ratio(
                sum(record.selection_correct is True for record in valid), selection_slots
            ),
            "decision_accuracy": ratio(sum(record.decision_correct for record in valid), len(trials)),
            "conditional_follow_correctness": ratio(
                sum(record.selection_correct is True for record in follows), len(follows)
            ),
            "follow_selection_counts": dict(Counter(str(record.selected_article) for record in follows)),
            "repeat_pair_agreement": ratio(sum(comparisons), len(comparisons)),
            "complete_repeat_groups": complete_groups,
            "constant_complete_repeat_groups": sum(
                len(valid_group) == len(group) and len(set(valid_group)) == 1
                for valid_group, group in zip(valid_groups, repeat_groups.values())
            ),
            "repeat_coverage": {
                "scheduled_groups": len(repeat_groups),
                "complete_groups": complete_groups,
                "incomplete_groups": len(repeat_groups) - complete_groups,
                "scheduled_pairs": scheduled_pairs,
                "comparable_valid_pairs": len(comparisons),
                "incomplete_pairs": scheduled_pairs - len(comparisons),
            },
            "token_usage": {
                key: {
                    "reported_count": sum(getattr(item, key) is not None for item in usage),
                    "sum": sum(getattr(item, key) or 0 for item in usage),
                }
                for key in ("input_tokens", "output_tokens", "total_tokens")
            },
            "message_counts": sorted(
                {trace_by_id[trial.prompt_trace_id].prompt_identity.message_count for trial in trials}
            ),
            "character_counts": sorted(
                {
                    trace_by_id[trial.prompt_trace_id].prompt_identity.character_count
                    for trial in trials
                }
            ),
        }

    def comparison(policy: str, members: list[Any]) -> dict[str, Any]:
        treatments = [[trial for trial in members if trial.prompt_variant_id == variant] for variant in VARIANTS]
        summaries = [measure(policy, trials) for trials in treatments]
        pairs: dict[str, dict[str, Any]] = defaultdict(dict)
        for trial in members:
            pairs[trial.prompt_treatment_pair_id][trial.prompt_variant_id] = result_by_id.get(
                (policy, trial.trial_id)
            )
        complete = [
            tuple(pair.get(variant) for variant in VARIANTS)
            for pair in pairs.values()
            if all(
                pair.get(variant) is not None and pair[variant].outcome.status == "ok"
                for variant in VARIANTS
            )
        ]
        transitions = Counter(
            (action_label(left.outcome), action_label(right.outcome)) for left, right in complete
        )

        def count_action(summary: dict[str, Any], label: str) -> int:
            counts = summary["action_counts"]
            if label == "FOLLOW":
                return sum(count for key, count in counts.items() if key.startswith("FOLLOW:"))
            if label == "outside-pair":
                return sum(
                    count for key, count in counts.items() if key not in {"STOP", "FOLLOW:3", "FOLLOW:41"}
                )
            return counts.get(label, 0)

        def matches_action(action: str, label: str) -> bool:
            if label == "FOLLOW":
                return action.startswith("FOLLOW:")
            if label == "outside-pair":
                return action not in {"STOP", "FOLLOW:3", "FOLLOW:41"}
            return action == label

        paired_count_changes = {}
        for label in ("STOP", "FOLLOW", "FOLLOW:3", "FOLLOW:41", "outside-pair"):
            baseline_count = sum(
                count
                for (baseline, _), count in transitions.items()
                if matches_action(baseline, label)
            )
            ablated_count = sum(
                count
                for (_, ablated), count in transitions.items()
                if matches_action(ablated, label)
            )
            paired_count_changes[label] = ablated_count - baseline_count
        raw_valid_count_changes = {
            label: count_action(summaries[1], label) - count_action(summaries[0], label)
            for label in ("STOP", "FOLLOW", "FOLLOW:3", "FOLLOW:41", "outside-pair")
        }
        differences = {}
        for metric in ("selection_accuracy", "decision_accuracy", "conditional_follow_correctness"):
            baseline, ablated = [summary[metric] for summary in summaries]
            differences[metric] = {
                "baseline": baseline,
                "ablated": ablated,
                "difference": ablated["value"] - baseline["value"]
                if ablated["value"] is not None and baseline["value"] is not None
                else None,
            }
        return {
            "variants": dict(zip(VARIANTS, summaries)),
            "scheduled_pairs": len(pairs),
            "complete_valid_pairs": len(complete),
            "incomplete_pairs": len(pairs) - len(complete),
            "action_agreement": ratio(
                sum(count for (left, right), count in transitions.items() if left == right),
                len(complete),
            ),
            "switch_count": sum(count for (left, right), count in transitions.items() if left != right),
            "transition_matrix": [
                {"baseline": left, "ablated": right, "count": count}
                for (left, right), count in sorted(transitions.items())
            ],
            "paired_count_changes": paired_count_changes,
            "raw_valid_count_changes": raw_valid_count_changes,
            "accuracy_differences": differences,
        }

    rows: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    roles: list[dict[str, Any]] = []
    macro: list[dict[str, Any]] = []
    for policy in manifest.policies:
        for phase in manifest.phases:
            phase_trials = [trial for trial in manifest.trials if trial.phase_id == phase]
            grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
            for trial in phase_trials:
                grouped[(trial.case_id, trial.arm_id)].append(trial)
            for (case_id, arm_id), members in grouped.items():
                identity = {
                    "policy": policy,
                    "phase_id": phase,
                    "case_id": case_id,
                    "question_id": case_by_id[case_id].question_id,
                    "arm_id": arm_id,
                    "case_role": members[0].case_role,
                }
                effects.append({**identity, **comparison(policy, members)})
                for variant in VARIANTS:
                    selected = [trial for trial in members if trial.prompt_variant_id == variant]
                    for batch_id in ("batch-0", "batch-1"):
                        rows.append(
                            {
                                **identity,
                                "variant_id": variant,
                                "batch_id": batch_id,
                                **measure(
                                    policy, [trial for trial in selected if trial.batch_id == batch_id]
                                ),
                            }
                        )
                    by_batch = {
                        (trial.batch_id, trial.repeat_id): result_by_id.get((policy, trial.trial_id))
                        for trial in selected
                    }
                    pairs = [
                        (by_batch.get(("batch-0", repeat)), by_batch.get(("batch-1", repeat)))
                        for repeat in range(3)
                    ]
                    complete = [
                        (left, right)
                        for left, right in pairs
                        if left and right and left.outcome.status == right.outcome.status == "ok"
                    ]
                    batches.append(
                        {
                            **identity,
                            "variant_id": variant,
                            "scheduled_pairs": 3,
                            "complete_valid_pairs": len(complete),
                            "agreement": ratio(
                                sum(
                                    action_label(left.outcome) == action_label(right.outcome)
                                    for left, right in complete
                                ),
                                len(complete),
                            ),
                            "incomplete_pairs": 3 - len(complete),
                        }
                    )
            if phase == PHASES[0]:
                for orientation in ("3-before-41", "41-before-3"):
                    members = [
                        trial
                        for trial in phase_trials
                        if (trial.presented_candidates.index(3) < trial.presented_candidates.index(41))
                        == (orientation == "3-before-41")
                    ]
                    strata.append({"policy": policy, "orientation": orientation, **comparison(policy, members)})
                for left, right in (
                    ("a1-pair-23-3-first", "a2-pair-23-41-first"),
                    ("a3-pair-34-3-first", "a4-pair-34-41-first"),
                ):
                    swaps = {}
                    cells = {"3-before-41": {}, "41-before-3": {}}
                    for variant in VARIANTS:
                        arms = [
                            [
                                trial
                                for trial in phase_trials
                                if trial.arm_id == arm and trial.prompt_variant_id == variant
                            ]
                            for arm in (left, right)
                        ]
                        counts = [measure(policy, trials) for trials in arms]
                        by_arm = [
                            {
                                (trial.batch_id, trial.repeat_id): result_by_id.get(
                                    (policy, trial.trial_id)
                                )
                                for trial in trials
                            }
                            for trials in arms
                        ]
                        valid_pairs = [
                            (left_record, by_arm[1].get(key))
                            for key, left_record in by_arm[0].items()
                            if left_record
                            and by_arm[1].get(key)
                            and left_record.outcome.status == by_arm[1][key].outcome.status == "ok"
                        ]
                        transitions = Counter(
                            (action_label(left_record.outcome), action_label(right_record.outcome))
                            for left_record, right_record in valid_pairs
                        )
                        swaps[variant] = {
                            "direction": "3-before-41 -> 41-before-3",
                            "scheduled_pairs": len(set(by_arm[0]) | set(by_arm[1])),
                            "complete_valid_pairs": len(valid_pairs),
                            "incomplete_pairs": len(set(by_arm[0]) | set(by_arm[1])) - len(valid_pairs),
                            "transition_matrix": [
                                {"from": left_label, "to": right_label, "count": count}
                                for (left_label, right_label), count in sorted(transitions.items())
                            ],
                            "agreement": ratio(
                                sum(
                                    count
                                    for (left_label, right_label), count in transitions.items()
                                    if left_label == right_label
                                ),
                                len(valid_pairs),
                            ),
                        }
                        for orientation, summary in zip(cells, counts):
                            cells[orientation][variant] = summary["decision_accuracy"]
                    effects_by_orientation = {}
                    for orientation, values in cells.items():
                        baseline, ablated = [values[variant]["value"] for variant in VARIANTS]
                        effects_by_orientation[orientation] = (
                            ablated - baseline
                            if baseline is not None and ablated is not None
                            else None
                        )
                    undefined = sum(
                        cell["value"] is None for values in cells.values() for cell in values.values()
                    )
                    interactions.append(
                        {
                            "policy": policy,
                            "arms": [left, right],
                            "variant_swap_tables": swaps,
                            "decision_accuracy_difference_in_differences": {
                                "cells": cells,
                                "effects": effects_by_orientation,
                                "undefined_components": undefined,
                                "value": effects_by_orientation["41-before-3"]
                                - effects_by_orientation["3-before-41"]
                                if not undefined
                                else None,
                            },
                        }
                    )
            else:
                for role in sorted({trial.case_role for trial in phase_trials}):
                    members = [trial for trial in phase_trials if trial.case_role == role]
                    roles.append(
                        {
                            "policy": policy,
                            "case_role": role,
                            "case_ids": list(dict.fromkeys(trial.case_id for trial in members)),
                            "per_case_effects": [
                                effect
                                for effect in effects
                                if effect["policy"] == policy
                                and effect["phase_id"] == phase
                                and effect["case_role"] == role
                            ],
                        }
                    )
                for variant in VARIANTS:
                    per_case = [
                        effect["variants"][variant]
                        for effect in effects
                        if effect["policy"] == policy and effect["phase_id"] == phase
                    ]
                    macro.append(
                        {
                            "policy": policy,
                            "variant_id": variant,
                            **{
                                metric: {
                                    "eligible_questions": len(per_case),
                                    "defined_questions": len(
                                        values := [
                                            effect[metric]["value"]
                                            for effect in per_case
                                            if effect[metric]["value"] is not None
                                        ]
                                    ),
                                    "undefined_questions": len(per_case) - len(values),
                                    "mean": sum(values) / len(values) if values else None,
                                }
                                for metric in ("decision_accuracy", "selection_accuracy")
                            },
                        }
                    )
    return {
        "artifact_schema_version": 3,
        "run_id": manifest.run_id,
        "status": _run_status(manifest, results),
        "coverage_by_policy": {
            policy: {
                "scheduled": len(manifest.trials),
                "observed": sum(result.policy == policy for result in results),
                "valid": sum(result.policy == policy and result.outcome.status == "ok" for result in results),
                "errors": sum(
                    result.policy == policy and result.outcome.status == "error" for result in results
                ),
                "missing": len(manifest.trials)
                - sum(result.policy == policy for result in results),
            }
            for policy in manifest.policies
        },
        "per_batch": rows,
        "per_case_arm_effects": effects,
        "batch_agreement": batches,
        "c1_orientation_effects": strata,
        "a_by_c_interactions": interactions,
        "c2_role_guards": roles,
        "c2_macro_accuracy": macro,
        "interpretation": "Removed example-group bundle effect only; not proof of memorization, causality of length, or corpus generalization",
    }


__all__ = [
    "action_label",
    "ratio",
    "summarize_prompt_ablation",
]
