"""Chấm điểm và tổng hợp STOP run thuần từ artifact đã lưu, không gọi model

Action correctness, selection correctness và decision correctness được giữ tách
biệt: FOLLOW sai Điều vẫn đúng action nhưng sai selection và sai decision. Mọi
metric đều mang tử số, mẫu số và nhận null khi mẫu số bằng 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts import GateCase, DecisionOutcome

from metrics import (
    action_key_from_outcome,
    decision_correctness,
    mean_metric,
    ratio_metric,
    selection_correctness,
    telemetry_summary,
)

from stop_policy_eval.contracts import (
    StopCaseMacro,
    StopLabel,
    StopLabelRates,
    StopPairedSummary,
    StopPolicyCase,
    StopPolicySummary,
    StopResult,
    StopRunManifest,
    StopStrataKind,
    StopStrataRow,
    StopSummary,
    hop_class_for,
)

STRATA_KIND_ORDER: tuple[StopStrataKind, ...] = ("case", "semantic_group", "source_hop", "split")
LABELS: tuple[StopLabel, ...] = ("follow", "stop")


def label_projection(case: StopPolicyCase) -> GateCase:
    """Chiếu một STOP case thành GateCase để dùng lại đúng scoring helper chung"""
    return GateCase(
        case_id=case.case_id,
        dataset_id=case.dataset_id,
        question_id=case.question_id,
        snapshot_id=case.snapshot_id,
        snapshot_hash=case.snapshot_hash,
        question=case.question,
        observation=case.observation,
        observation_hash=case.observation_hash,
        candidates=list(case.candidates),
        source_hop=case.source_hop,
        source_run_id=case.source_run_id,
        source_policy=case.source_policy,
        expected_action=case.expected_action,
        acceptable_dieu=set(case.acceptable_dieu),
        label_reason=case.label_reason,
        label_status=case.label_status,
        split=case.split,
        fewshot_overlap=case.fewshot_overlap,
    )


def derive_result_score(
    case: StopPolicyCase,
    presented_candidates: list[int],
    outcome: DecisionOutcome,
) -> dict[str, Any]:
    """Tính lại vị trí, selection và decision correctness từ outcome và nhãn đã duyệt"""
    selected_article: int | None = None
    selected_position: int | None = None
    if outcome.status == "ok" and outcome.decision is not None and not outcome.decision.stop:
        selected_article = outcome.decision.dieu
        if selected_article not in presented_candidates:
            raise ValueError(
                f"a valid follow selected {selected_article} outside {presented_candidates}"
            )
        selected_position = presented_candidates.index(selected_article) + 1
    projection = label_projection(case)
    return {
        "selected_article": selected_article,
        "selected_position": selected_position,
        "selection_correct": selection_correctness(projection, outcome),
        "decision_correct": decision_correctness(projection, outcome),
    }


@dataclass
class _Slots:
    """Bộ đếm coverage và action của một tập slot đã schedule"""

    scheduled: int = 0
    valid: int = 0
    errors: int = 0
    missing: int = 0
    scheduled_by_label: dict[str, int] = field(
        default_factory=lambda: {label: 0 for label in LABELS}
    )
    valid_by_label: dict[str, int] = field(
        default_factory=lambda: {label: 0 for label in LABELS}
    )
    errors_by_label: dict[str, int] = field(
        default_factory=lambda: {label: 0 for label in LABELS}
    )
    missing_by_label: dict[str, int] = field(
        default_factory=lambda: {label: 0 for label in LABELS}
    )
    follow_actions: int = 0
    stops_in_follow: int = 0
    stops_in_stop: int = 0
    follows_in_stop: int = 0
    action_correct: int = 0
    selection_correct: int = 0
    selection_scheduled: int = 0
    decision_correct: int = 0
    valid_results: list[StopResult] = field(default_factory=list)


def _accumulate(slots: _Slots, case: StopPolicyCase, result: StopResult | None) -> None:
    """Cộng một slot vào bộ đếm, không bao giờ biến lỗi hay thiếu thành STOP"""
    label = case.expected_action
    if label not in LABELS:
        raise ValueError(f"{case.case_id}: only follow and stop labels can be scheduled")
    slots.scheduled += 1
    slots.scheduled_by_label[label] += 1
    if label == "follow":
        slots.selection_scheduled += 1

    if result is None:
        slots.missing += 1
        slots.missing_by_label[label] += 1
        return
    if result.outcome.status != "ok" or result.outcome.decision is None:
        slots.errors += 1
        slots.errors_by_label[label] += 1
        return

    slots.valid += 1
    slots.valid_by_label[label] += 1
    slots.valid_results.append(result)
    score = derive_result_score(case, list(case.candidates), result.outcome)
    is_follow = not result.outcome.decision.stop
    if label == "follow":
        if is_follow:
            slots.follow_actions += 1
            slots.action_correct += 1
        else:
            slots.stops_in_follow += 1
        if score["selection_correct"]:
            slots.selection_correct += 1
    else:
        if is_follow:
            slots.follows_in_stop += 1
        else:
            slots.stops_in_stop += 1
            slots.action_correct += 1
    if score["decision_correct"]:
        slots.decision_correct += 1


def _label_rates(slots: _Slots) -> StopLabelRates:
    """Action rate hợp lệ theo từng lớp nhãn"""
    follow_valid = slots.valid_by_label["follow"]
    stop_valid = slots.valid_by_label["stop"]
    return StopLabelRates(
        follow_accuracy=ratio_metric(slots.follow_actions, follow_valid),
        false_stop_rate=ratio_metric(slots.stops_in_follow, follow_valid),
        stop_accuracy=ratio_metric(slots.stops_in_stop, stop_valid),
        over_hop_rate=ratio_metric(slots.follows_in_stop, stop_valid),
        follow_valid=follow_valid,
        stop_valid=stop_valid,
    )


def _case_macro(per_case: list[_Slots]) -> StopCaseMacro:
    """Lấy giá trị định nghĩa được của từng case rồi lấy mean giữa các case"""
    eligible = len(per_case)
    follow_values: list[float] = []
    stop_values: list[float] = []
    action_values: list[float] = []
    selection_values: list[float] = []
    decision_values: list[float] = []
    for slots in per_case:
        follow_valid = slots.valid_by_label["follow"]
        if follow_valid:
            follow_values.append(slots.follow_actions / follow_valid)
        stop_valid = slots.valid_by_label["stop"]
        if stop_valid:
            stop_values.append(slots.stops_in_stop / stop_valid)
        action_values.append(slots.action_correct / slots.scheduled)
        if slots.selection_scheduled:
            selection_values.append(slots.selection_correct / slots.selection_scheduled)
        decision_values.append(slots.decision_correct / slots.scheduled)

    def macro(values: list[float]) -> Any:
        return mean_metric(values, eligible=eligible, excluded=eligible - len(values))

    return StopCaseMacro(
        case_count=eligible,
        follow_accuracy=macro(follow_values),
        stop_accuracy=macro(stop_values),
        action_accuracy=macro(action_values),
        selection_accuracy=macro(selection_values),
        decision_accuracy=macro(decision_values),
    )


def _policy_summary(slots: _Slots, per_case: list[_Slots]) -> StopPolicySummary:
    return StopPolicySummary(
        scheduled=slots.scheduled,
        valid=slots.valid,
        errors=slots.errors,
        missing=slots.missing,
        action_accuracy=ratio_metric(slots.action_correct, slots.scheduled),
        selection_accuracy=ratio_metric(slots.selection_correct, slots.selection_scheduled),
        decision_accuracy=ratio_metric(slots.decision_correct, slots.scheduled),
        label_rates=_label_rates(slots),
        error_rate={
            label: ratio_metric(
                slots.errors_by_label[label], slots.scheduled_by_label[label]
            )
            for label in LABELS
        },
        missing_rate={
            label: ratio_metric(
                slots.missing_by_label[label], slots.scheduled_by_label[label]
            )
            for label in LABELS
        },
        case_macro=_case_macro(per_case),
        telemetry=telemetry_summary(slots.valid_results),
    )


def _stratum_keys(case: StopPolicyCase) -> list[tuple[StopStrataKind, str]]:
    keys: list[tuple[StopStrataKind, str]] = [("case", case.case_id)]
    if case.semantic_group_id is not None:
        keys.append(("semantic_group", case.semantic_group_id))
    keys.append(("source_hop", hop_class_for(case.source_hop)))
    keys.append(("split", case.split))
    return keys


def _strata_rows(
    manifest: StopRunManifest,
    case_by_id: dict[str, StopPolicyCase],
    result_by_key: dict[tuple[str, str], StopResult],
) -> list[StopStrataRow]:
    rows: list[StopStrataRow] = []
    for policy in manifest.policies:
        groups: dict[tuple[StopStrataKind, str], list[tuple[StopPolicyCase, StopResult | None]]] = {}
        for trial in manifest.trials:
            case = case_by_id[trial.case_id]
            result = result_by_key.get((policy, trial.trial_id))
            for kind, value in _stratum_keys(case):
                groups.setdefault((kind, value), []).append((case, result))
        for kind, value in sorted(
            groups, key=lambda item: (STRATA_KIND_ORDER.index(item[0]), item[1])
        ):
            slots = _Slots()
            for case, result in groups[(kind, value)]:
                _accumulate(slots, case, result)
            rows.append(
                StopStrataRow(
                    kind=kind,
                    value=value,
                    policy=policy,
                    scheduled=slots.scheduled,
                    valid=slots.valid,
                    errors=slots.errors,
                    missing=slots.missing,
                    action_accuracy=ratio_metric(slots.action_correct, slots.scheduled),
                    selection_accuracy=ratio_metric(
                        slots.selection_correct, slots.selection_scheduled
                    ),
                    decision_accuracy=ratio_metric(slots.decision_correct, slots.scheduled),
                )
            )
    return rows


def _paired_summary(
    manifest: StopRunManifest,
    case_by_id: dict[str, StopPolicyCase],
    result_by_key: dict[tuple[str, str], StopResult],
) -> StopPairedSummary | None:
    """So sánh first với llm trên cùng trial ID, tách agreement khỏi correctness"""
    policies = list(manifest.policies)
    if set(policies) != {"first", "llm"}:
        return None

    valid_pairs = 0
    incomplete_pairs = 0
    agreement = 0
    a_wins = 0
    b_wins = 0
    ties = 0
    different: list[str] = []
    for trial in manifest.trials:
        case = case_by_id[trial.case_id]
        first = result_by_key.get(("first", trial.trial_id))
        llm = result_by_key.get(("llm", trial.trial_id))
        if (
            first is None
            or llm is None
            or first.outcome.status != "ok"
            or llm.outcome.status != "ok"
            or first.outcome.decision is None
            or llm.outcome.decision is None
        ):
            incomplete_pairs += 1
            continue
        valid_pairs += 1
        if action_key_from_outcome(first.outcome) == action_key_from_outcome(llm.outcome):
            agreement += 1
        else:
            different.append(trial.trial_id)
        first_correct = derive_result_score(
            case, list(case.candidates), first.outcome
        )["decision_correct"]
        llm_correct = derive_result_score(case, list(case.candidates), llm.outcome)[
            "decision_correct"
        ]
        if first_correct and not llm_correct:
            a_wins += 1
        elif llm_correct and not first_correct:
            b_wins += 1
        else:
            ties += 1

    return StopPairedSummary(
        policies=policies,
        scheduled_pairs=len(manifest.trials),
        valid_pairs=valid_pairs,
        incomplete_pairs=incomplete_pairs,
        action_agreement=ratio_metric(agreement, valid_pairs),
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        different_trial_ids=different,
    )


def _index_run(
    manifest: StopRunManifest,
    cases: list[StopPolicyCase],
    results: list[StopResult],
) -> tuple[dict[str, StopPolicyCase], dict[tuple[str, str], StopResult]]:
    """Lập chỉ mục case và result, từ chối key trùng hoặc lệch schedule"""
    case_by_id: dict[str, StopPolicyCase] = {}
    for case in cases:
        if case.case_id in case_by_id:
            raise ValueError(f"cases contain duplicate case_id values: {case.case_id}")
        case_by_id[case.case_id] = case
    trial_by_id = {trial.trial_id: trial for trial in manifest.trials}
    if len(trial_by_id) != len(manifest.trials):
        raise ValueError("manifest contains duplicate trial_id values")
    for trial in manifest.trials:
        case = case_by_id.get(trial.case_id)
        if case is None:
            raise ValueError(f"trial uses an unknown case: {trial.case_id}")
        if case.expected_action not in LABELS:
            raise ValueError(f"unlabeled case in the schedule: {case.case_id}")

    result_by_key: dict[tuple[str, str], StopResult] = {}
    for result in results:
        if result.policy not in manifest.policies:
            raise ValueError(f"result uses an unscheduled policy: {result.policy}")
        trial = trial_by_id.get(result.trial_id)
        if trial is None:
            raise ValueError(f"result uses an unknown trial: {result.trial_id}")
        if result.run_id != manifest.run_id:
            raise ValueError(f"result uses a different run_id: {result.trial_id}")
        if result.input_trace_id != trial.input_trace_id:
            raise ValueError(f"result references a different input trace: {result.trial_id}")
        key = (result.policy, result.trial_id)
        if key in result_by_key:
            raise ValueError(f"duplicate result key: {key}")
        result_by_key[key] = result
    return case_by_id, result_by_key


def summarize_stop_run(
    manifest: StopRunManifest,
    cases: list[StopPolicyCase],
    results: list[StopResult],
) -> StopSummary:
    """Tính lại mọi metric STOP từ schedule, nhãn đã duyệt và outcome đã lưu

    Params:
    - manifest: manifest của run, quyết định policy, split và schedule
    - cases: case đã duyệt mà schedule tham chiếu
    - results: result compact đã reload và verify
    """
    case_by_id, result_by_key = _index_run(manifest, cases, results)

    by_policy: dict[str, StopPolicySummary] = {}
    total_scheduled = 0
    total_valid = 0
    total_errors = 0
    total_missing = 0
    for policy in manifest.policies:
        slots = _Slots()
        per_case: dict[str, _Slots] = {}
        case_order: list[str] = []
        for trial in manifest.trials:
            case = case_by_id[trial.case_id]
            if case.case_id not in per_case:
                per_case[case.case_id] = _Slots()
                case_order.append(case.case_id)
            result = result_by_key.get((policy, trial.trial_id))
            _accumulate(slots, case, result)
            _accumulate(per_case[case.case_id], case, result)
        by_policy[policy] = _policy_summary(slots, [per_case[cid] for cid in case_order])
        total_scheduled += slots.scheduled
        total_valid += slots.valid
        total_errors += slots.errors
        total_missing += slots.missing

    return StopSummary(
        schema_version=manifest.schema_version,
        experiment="stop-policy",
        run_id=manifest.run_id,
        variant=manifest.variant,
        split=manifest.split,
        policies=list(manifest.policies),
        scheduled=total_scheduled,
        valid=total_valid,
        errors=total_errors,
        missing=total_missing,
        by_policy=by_policy,
        strata=_strata_rows(manifest, case_by_id, result_by_key),
        readiness=manifest.readiness,
        paired=_paired_summary(manifest, case_by_id, result_by_key),
        exclusions=list(manifest.exclusions),
    )


__all__ = [
    "derive_result_score",
    "label_projection",
    "summarize_stop_run",
]
