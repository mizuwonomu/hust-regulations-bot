"""Tính lại metric của intervention A từ artifact bền vững, không gọi model."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from contracts import DecisionOutcome, GateCase
from metrics import action_key_from_outcome, ratio_metric

from diagnostic_subexp.candidate_pair_41_3_position.contracts import (
    CandidateAccuracyRow,
    CandidateActionAgreement,
    CandidateActionCountRow,
    CandidateBatchAgreementRow,
    CandidateInputTrace,
    CandidateManifest,
    CandidateOutputs,
    CandidatePairSelection,
    CandidateRatioSummary,
    CandidateRepeatConsistencyRow,
    CandidateSummary,
    CandidateTrial,
    DIAGNOSTIC_KIND,
    PAIR_ARTICLES,
)
from diagnostic_subexp.shared.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    EXPERIMENT_NAME,
    DiagnosticResult,
)

SlotKey = tuple[str, str]


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


def _action_sort_key(label: str) -> tuple[int, int, str]:
    if label == "STOP":
        return (0, 0, label)
    if label.startswith("FOLLOW:"):
        return (1, int(label.split(":", 1)[1]), label)
    return (2, 0, label)


def _label_set(labels: list[str]) -> list[str]:
    return sorted(set(labels), key=_action_sort_key)


def _slot_index(
    manifest: CandidateManifest,
    results: list[DiagnosticResult],
) -> tuple[list[tuple[str, CandidateTrial]], dict[SlotKey, DiagnosticResult]]:
    slots = [(policy, trial) for policy in manifest.policies for trial in manifest.trials]
    by_key = {(result.policy, result.trial_id): result for result in results}
    return slots, by_key


def _coverage(
    slots: list[tuple[str, CandidateTrial]],
    by_key: dict[SlotKey, DiagnosticResult],
) -> tuple[int, int, int]:
    valid = sum(
        1
        for policy, trial in slots
        if (result := by_key.get((policy, trial.trial_id))) is not None
        and result.outcome.status == "ok"
    )
    errors = sum(
        1
        for policy, trial in slots
        if (result := by_key.get((policy, trial.trial_id))) is not None
        and result.outcome.status == "error"
    )
    missing = len(slots) - valid - errors
    return valid, errors, missing


def _ratio_rows(
    slots: list[tuple[str, CandidateTrial]],
    by_key: dict[SlotKey, DiagnosticResult],
    case: GateCase,
) -> list[CandidateAccuracyRow]:
    buckets: dict[tuple[str, str], list[tuple[str, CandidateTrial]]] = defaultdict(list)
    for policy, trial in slots:
        buckets[(policy, trial.arm_id)].append((policy, trial))
    rows: list[CandidateAccuracyRow] = []
    for (policy, arm_id), members in sorted(buckets.items()):
        valid, errors, missing = _coverage(members, by_key)
        selection_numerator = 0
        decision_numerator = 0
        for slot_policy, trial in members:
            result = by_key.get((slot_policy, trial.trial_id))
            if result is None or result.outcome.status != "ok":
                continue
            if case.expected_action == "follow" and result.selection_correct:
                selection_numerator += 1
            if result.decision_correct:
                decision_numerator += 1
        selection_denominator = len(members) if case.expected_action == "follow" else 0
        rows.append(
            CandidateAccuracyRow(
                policy=policy,
                arm_id=arm_id,
                case_id=case.case_id,
                scheduled=len(members),
                valid=valid,
                errors=errors,
                missing=missing,
                selection_accuracy=ratio_metric(selection_numerator, selection_denominator),
                decision_accuracy=ratio_metric(decision_numerator, len(members)),
            )
        )
    return rows


def _action_rows(
    slots: list[tuple[str, CandidateTrial]],
    by_key: dict[SlotKey, DiagnosticResult],
) -> list[CandidateActionCountRow]:
    rows: list[CandidateActionCountRow] = []
    pooled: dict[tuple[str, str], list[tuple[str, CandidateTrial]]] = defaultdict(list)
    detailed: dict[tuple[str, str, str, int], list[tuple[str, CandidateTrial]]] = defaultdict(list)
    for policy, trial in slots:
        pooled[(policy, trial.arm_id)].append((policy, trial))
        detailed[(policy, trial.arm_id, trial.batch_id, trial.repeat_id)].append((policy, trial))

    for (policy, arm_id), members in sorted(pooled.items()):
        rows.append(_action_row(policy, arm_id, None, None, members, by_key))
    for (policy, arm_id, batch_id, repeat_id), members in sorted(
        detailed.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])
    ):
        rows.append(_action_row(policy, arm_id, batch_id, repeat_id, members, by_key))
    return rows


def _action_row(
    policy: str,
    arm_id: str,
    batch_id: str | None,
    repeat_id: int | None,
    members: list[tuple[str, CandidateTrial]],
    by_key: dict[SlotKey, DiagnosticResult],
) -> CandidateActionCountRow:
    counts: dict[str, int] = defaultdict(int)
    for slot_policy, trial in members:
        result = by_key.get((slot_policy, trial.trial_id))
        label = action_label(result.outcome) if result is not None else None
        if label is not None:
            counts[label] += 1
    valid, errors, missing = _coverage(members, by_key)
    return CandidateActionCountRow(
        policy=policy,
        arm_id=arm_id,
        batch_id=batch_id,
        repeat_id=repeat_id,
        scheduled=len(members),
        valid=valid,
        errors=errors,
        missing=missing,
        action_counts=dict(sorted(counts.items(), key=lambda item: _action_sort_key(item[0]))),
    )


def _repeat_rows(
    slots: list[tuple[str, CandidateTrial]],
    by_key: dict[SlotKey, DiagnosticResult],
) -> list[CandidateRepeatConsistencyRow]:
    groups: dict[tuple[str, str, str, str], list[tuple[str, CandidateTrial]]] = defaultdict(list)
    for policy, trial in slots:
        groups[(policy, trial.arm_id, trial.batch_id, trial.request_fingerprint)].append(
            (policy, trial)
        )
    rows: list[CandidateRepeatConsistencyRow] = []
    for (policy, arm_id, batch_id, fingerprint), members in sorted(groups.items()):
        labels = [
            action_label(by_key.get((slot_policy, trial.trial_id)).outcome)
            for slot_policy, trial in members
            if (slot_policy, trial.trial_id) in by_key
            and by_key[(slot_policy, trial.trial_id)].outcome.status == "ok"
        ]
        valid = len(labels)
        pairs = list(combinations(range(valid), 2))
        matching = sum(1 for left, right in pairs if labels[left] == labels[right])
        rows.append(
            CandidateRepeatConsistencyRow(
                policy=policy,
                arm_id=arm_id,
                batch_id=batch_id,
                request_fingerprint=fingerprint,
                scheduled=len(members),
                valid=valid,
                matching_pairs=matching,
                valid_pairs=len(pairs),
                consistency=ratio_metric(matching, len(pairs)),
            )
        )
    return rows


def _batch_rows(
    slots: list[tuple[str, CandidateTrial]],
    by_key: dict[SlotKey, DiagnosticResult],
) -> list[CandidateBatchAgreementRow]:
    groups: dict[tuple[str, str, str, int, str], dict[str, DiagnosticResult | None]] = defaultdict(dict)
    for policy, trial in slots:
        groups[
            (policy, trial.arm_id, trial.diagnostic_kind, trial.repeat_id, trial.request_fingerprint)
        ][trial.batch_id] = by_key.get((policy, trial.trial_id))
    rows: list[CandidateBatchAgreementRow] = []
    for (policy, arm_id, _kind, repeat_id, fingerprint), by_batch in sorted(groups.items()):
        batch_ids = sorted(by_batch)
        pairs = list(combinations(batch_ids, 2))
        valid_pairs = 0
        matching = 0
        switches = 0
        for left_batch, right_batch in pairs:
            left_result = by_batch[left_batch]
            right_result = by_batch[right_batch]
            if left_result is None or right_result is None:
                continue
            left = action_label(left_result.outcome)
            right = action_label(right_result.outcome)
            if left is None or right is None:
                continue
            valid_pairs += 1
            if left == right:
                matching += 1
            else:
                switches += 1
        rows.append(
            CandidateBatchAgreementRow(
                policy=policy,
                arm_id=arm_id,
                request_fingerprint=fingerprint,
                repeat_id=repeat_id,
                scheduled=len(pairs),
                valid=valid_pairs,
                agreement=ratio_metric(matching, valid_pairs),
                switch_count=switches,
            )
        )
    return rows


def _comparison(
    policy_groups: dict[tuple[str, str, int], tuple[DiagnosticResult | None, DiagnosticResult | None]],
    *,
    comparison_id: str,
    policy: str,
    left_arm: str,
    right_arm: str,
) -> CandidateActionAgreement:
    labels: list[str] = []
    transition: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    valid = 0
    switches = 0
    for left, right in policy_groups.values():
        left_label = action_label(left.outcome) if left is not None else None
        right_label = action_label(right.outcome) if right is not None else None
        if left_label is None or right_label is None:
            continue
        valid += 1
        labels.extend([left_label, right_label])
        transition[left_label][right_label] += 1
        if left_label != right_label:
            switches += 1
    keys = _label_set(labels)
    matrix = {
        left_key: {right_key: int(transition[left_key][right_key]) for right_key in keys}
        for left_key in keys
    }
    scheduled = len(policy_groups)
    return CandidateActionAgreement(
        comparison_id=comparison_id,
        policy=policy,
        left_arm=left_arm,
        right_arm=right_arm,
        scheduled=scheduled,
        valid=valid,
        incomplete=scheduled - valid,
        agreement=ratio_metric(valid - switches, valid),
        switch_count=switches,
        action_keys=keys,
        transition=matrix,
    )


def _pair_groups(
    slots: list[tuple[str, CandidateTrial]],
    by_key: dict[SlotKey, DiagnosticResult],
    *,
    left_arm: str,
    right_arm: str,
) -> dict[str, dict[tuple[str, str, int], tuple[DiagnosticResult | None, DiagnosticResult | None]]]:
    grouped: dict[str, dict[tuple[str, str, int], dict[str, DiagnosticResult | None]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for policy, trial in slots:
        if trial.arm_id not in {left_arm, right_arm}:
            continue
        grouped[policy][(trial.batch_id, trial.diagnostic_kind, trial.repeat_id)][trial.arm_id] = (
            by_key.get((policy, trial.trial_id))
        )
    output: dict[
        str,
        dict[tuple[str, str, int], tuple[DiagnosticResult | None, DiagnosticResult | None]],
    ] = defaultdict(dict)
    for policy, by_group in grouped.items():
        for key, by_arm in by_group.items():
            output[policy][key] = (by_arm.get(left_arm), by_arm.get(right_arm))
    return output


def _matched_pair_ids(manifest: CandidateManifest) -> dict[str, list[CandidateTrial]]:
    pairs: dict[str, list[CandidateTrial]] = defaultdict(list)
    for trial in manifest.trials:
        if trial.matched_pair_id is not None:
            pairs[trial.matched_pair_id].append(trial)
    return pairs


def _arm_with_orientation(
    arms: list[str],
    traces: dict[str, CandidateInputTrace],
    orientation: str,
) -> str:
    for arm_id in arms:
        traces_for_arm = [trace for trace in traces.values() if trace.arm_id == arm_id]
        if traces_for_arm and traces_for_arm[0].relative_pair_order == orientation:
            return arm_id
    raise ValueError(f"no arm declares relative pair order {orientation!r}")


def _candidate_outputs(
    slots: list[tuple[str, CandidateTrial]],
    by_key: dict[SlotKey, DiagnosticResult],
    manifest: CandidateManifest,
    traces: dict[str, CandidateInputTrace],
) -> CandidateOutputs:
    buckets: dict[tuple[str, str], list[CandidateTrial]] = defaultdict(list)
    for policy, trial in slots:
        buckets[(policy, trial.arm_id)].append(trial)

    pair_selections: list[CandidatePairSelection] = []
    for (policy, arm_id), members in sorted(buckets.items()):
        article_3 = article_41 = earlier = pair_count = stops = outside = 0
        position_counts: dict[str, int] = defaultdict(int)
        trace = traces[members[0].input_trace_id]
        for trial in members:
            result = by_key.get((policy, trial.trial_id))
            if result is None or result.outcome.status != "ok":
                continue
            if result.outcome.decision is not None and result.outcome.decision.stop:
                stops += 1
                continue
            selected = result.selected_article
            if selected == 3:
                article_3 += 1
            elif selected == 41:
                article_41 += 1
            else:
                outside += 1
                continue
            pair_count += 1
            positions = list(trace.pair_positions)
            if positions and result.selected_position in positions:
                position_counts[str(result.selected_position)] += 1
            earlier_index = min(trace.presented_candidates.index(article) for article in PAIR_ARTICLES)
            if result.selected_position == earlier_index + 1:
                earlier += 1
        pair_selections.append(
            CandidatePairSelection(
                policy=policy,
                arm_id=arm_id,
                article_3_count=article_3,
                article_41_count=article_41,
                earlier_of_pair_count=earlier,
                pair_selection_count=pair_count,
                earlier_of_pair=ratio_metric(earlier, pair_count),
                absolute_position_counts=dict(sorted(position_counts.items())),
                stop_count=stops,
                outside_pair_follow_count=outside,
            )
        )

    matched_swaps: list[CandidateActionAgreement] = []
    for pair_id, members in sorted(_matched_pair_ids(manifest).items()):
        arms = sorted({trial.arm_id for trial in members})
        if len(arms) != 2:
            continue
        left = _arm_with_orientation(arms, traces, "3-41")
        right = _arm_with_orientation(arms, traces, "41-3")
        grouped = _pair_groups(slots, by_key, left_arm=left, right_arm=right)
        for policy, groups in sorted(grouped.items()):
            matched_swaps.append(
                _comparison(
                    groups,
                    comparison_id=f"{left}<->{right}",
                    policy=policy,
                    left_arm=left,
                    right_arm=right,
                )
            )
    return CandidateOutputs(pair_selections=pair_selections, matched_swaps=matched_swaps)


def summarize_candidate_run(
    manifest: CandidateManifest,
    cases: list[GateCase],
    traces: list[CandidateInputTrace],
    results: list[DiagnosticResult],
) -> CandidateSummary:
    """Tính lại mọi slice của A từ artifact đã lưu."""
    if not isinstance(manifest, CandidateManifest):
        raise TypeError("manifest must be a CandidateManifest")
    case_by_id = {case.case_id: case for case in cases}
    case = case_by_id.get(manifest.case_id)
    if case is None:
        raise ValueError("summary requires the frozen source case")
    trace_by_id = {trace.input_trace_id: trace for trace in traces}
    slots, by_key = _slot_index(manifest, results)
    valid, errors, missing = _coverage(slots, by_key)

    by_policy: dict[str, CandidateRatioSummary] = {}
    for policy in manifest.policies:
        policy_slots = [(slot_policy, trial) for slot_policy, trial in slots if slot_policy == policy]
        policy_valid, policy_errors, policy_missing = _coverage(policy_slots, by_key)
        selection_numerator = 0
        decision_numerator = 0
        for slot_policy, trial in policy_slots:
            result = by_key.get((slot_policy, trial.trial_id))
            if result is None or result.outcome.status != "ok":
                continue
            if case.expected_action == "follow" and result.selection_correct:
                selection_numerator += 1
            if result.decision_correct:
                decision_numerator += 1
        selection_denominator = len(policy_slots) if case.expected_action == "follow" else 0
        by_policy[policy] = CandidateRatioSummary(
            scheduled=len(policy_slots),
            valid=policy_valid,
            errors=policy_errors,
            missing=policy_missing,
            selection_accuracy=ratio_metric(selection_numerator, selection_denominator),
            decision_accuracy=ratio_metric(decision_numerator, len(policy_slots)),
        )

    return CandidateSummary(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        experiment=EXPERIMENT_NAME,
        run_id=manifest.run_id,
        suite_id=manifest.suite_id,
        diagnostic_kind=DIAGNOSTIC_KIND,
        case_id=manifest.case_id,
        policies=list(manifest.policies),
        independent_question_count=manifest.independent_question_count,
        scheduled=len(slots),
        valid=valid,
        errors=errors,
        missing=missing,
        by_policy=by_policy,
        action_counts=_action_rows(slots, by_key),
        accuracy=_ratio_rows(slots, by_key, case),
        repeat_consistency=_repeat_rows(slots, by_key),
        batch_agreement=_batch_rows(slots, by_key),
        candidate_relative_position=_candidate_outputs(slots, by_key, manifest, trace_by_id),
        exclusions=list(manifest.exclusions),
    )


__all__ = [
    "action_label",
    "summarize_candidate_run",
]
