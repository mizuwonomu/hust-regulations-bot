"""Tổng hợp permutation deterministic từ schedule và result compact đã lưu."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

try:
    from contracts import (
        CaseMetrics,
        ConditionSummary,
        ConsistencySummary,
        GateCase,
        GoldPositionCase,
        GoldPositionGroup,
        IdenticalInputGroup,
        MeanMetric,
        PairedConditionSummary,
        PairedPolicySummary,
        PermutationCondition,
        PermutationSummary,
        PolicyName,
        PolicyPermutationSummary,
        PositionCohort,
        PositionDistribution,
        QuestionHopMetrics,
        QuestionKey,
        QuestionMacroSummary,
        ResultRecord,
        RunManifest,
        Trial,
    )
    from metrics import (
        action_key_from_outcome,
        decision_correctness,
        mean_metric,
        ratio_metric,
        selection_correctness,
        telemetry_summary,
        validate_result_set,
    )
except ModuleNotFoundError:
    from .contracts import (
        CaseMetrics,
        ConditionSummary,
        ConsistencySummary,
        GateCase,
        GoldPositionCase,
        GoldPositionGroup,
        IdenticalInputGroup,
        MeanMetric,
        PairedConditionSummary,
        PairedPolicySummary,
        PermutationCondition,
        PermutationSummary,
        PolicyName,
        PolicyPermutationSummary,
        PositionCohort,
        PositionDistribution,
        QuestionHopMetrics,
        QuestionKey,
        QuestionMacroSummary,
        ResultRecord,
        RunManifest,
        Trial,
    )
    from .metrics import (
        action_key_from_outcome,
        decision_correctness,
        mean_metric,
        ratio_metric,
        selection_correctness,
        telemetry_summary,
        validate_result_set,
    )


SEED_ORDER_CONDITIONS = frozenset({"seed-order", "combined"})
MIN_MULTI_CANDIDATES = 2
# Pair dùng action key để tách consistency khỏi vị trí candidate


@dataclass(frozen=True)
class _ConsistencyAggregation:
    """Giữ summary pooled và mean consistency theo từng case."""

    summary: ConsistencySummary
    by_case: dict[str, MeanMetric]


def _pair_counts(keys: Sequence[Any | None]) -> tuple[int, int, int]:
    """Đếm pair khớp, hợp lệ và đã schedule từ tần suất action key."""
    valid_keys = [key for key in keys if key is not None]
    frequencies: defaultdict[Any, int] = defaultdict(int)
    for key in valid_keys:
        frequencies[key] += 1
    matching = sum(count * (count - 1) // 2 for count in frequencies.values())
    valid_pairs = len(valid_keys) * (len(valid_keys) - 1) // 2
    scheduled_pairs = len(keys) * (len(keys) - 1) // 2
    return matching, valid_pairs, scheduled_pairs


def _consistency_summary(
    groups: dict[tuple[str, Any], list[Trial]],
    result_by_key: dict[tuple[str, str], ResultRecord],
    policy: PolicyName,
    *,
    scheduled_cases: set[str],
    min_candidate_count: int | None = None,
) -> _ConsistencyAggregation:
    matching = 0
    valid_pairs = 0
    scheduled_pairs = 0
    eligible_groups = 0
    excluded_groups = 0
    eligible_cases: set[str] = set()
    case_group_values: dict[str, list[float]] = defaultdict(list)
    case_eligible_groups: defaultdict[str, int] = defaultdict(int)

    for group_key in sorted(groups, key=lambda item: (item[0], repr(item[1]))):
        case_id = group_key[0]
        group_trials = groups[group_key]
        if min_candidate_count is not None and len(group_trials[0].candidate_order) < min_candidate_count:
            excluded_groups += 1
            continue
        if len(group_trials) < 2:
            excluded_groups += 1
            continue
        eligible_groups += 1
        eligible_cases.add(case_id)
        case_eligible_groups[case_id] += 1
        keys = [
            action_key_from_outcome(result_by_key[(policy, trial.trial_id)].outcome)
            if (policy, trial.trial_id) in result_by_key
            else None
            for trial in group_trials
        ]
        group_matching, group_valid, group_scheduled = _pair_counts(keys)
        matching += group_matching
        valid_pairs += group_valid
        scheduled_pairs += group_scheduled
        if group_valid:
            case_group_values[case_id].append(group_matching / group_valid)

    by_case: dict[str, MeanMetric] = {}
    for case_id in sorted(scheduled_cases):
        values = case_group_values.get(case_id, [])
        eligible = case_eligible_groups.get(case_id, 0)
        by_case[case_id] = mean_metric(
            values,
            eligible=eligible,
            excluded=eligible - len(values),
        )
    case_values = [metric.value for metric in by_case.values() if metric.value is not None]
    summary = ConsistencySummary(
        matching_pairs=matching,
        scheduled_pairs=scheduled_pairs,
        valid_pairs=valid_pairs,
        pooled=ratio_metric(matching, valid_pairs),
        eligible_groups=eligible_groups,
        excluded_groups=excluded_groups,
        eligible_cases=len(eligible_cases),
        excluded_cases=len(scheduled_cases - eligible_cases),
        case_mean=mean_metric(
            case_values,
            eligible=len(scheduled_cases),
            excluded=len(scheduled_cases) - len(case_values),
        ),
    )
    return _ConsistencyAggregation(summary=summary, by_case=by_case)


def _case_counters(
    case: GateCase,
    trials: list[Trial],
    result_by_key: dict[tuple[str, str], ResultRecord],
    policy: PolicyName,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    scheduled = len(trials)
    valid = errors = missing = 0
    selection_numerator = selection_denominator = 0
    decision_numerator = decision_denominator = 0
    first_numerator = first_denominator = 0

    for trial in trials:
        result = result_by_key.get((policy, trial.trial_id))
        if result is None:
            missing += 1
        else:
            if result.outcome.status == "error":
                errors += 1
            else:
                valid += 1
            if selection_correctness(case, result.outcome) is True:
                selection_numerator += 1
            if decision_correctness(case, result.outcome):
                decision_numerator += 1
            if result.outcome.status == "ok" and len(trial.candidate_order) >= MIN_MULTI_CANDIDATES:
                first_denominator += 1
                if result.selected_position == 1:
                    first_numerator += 1
        if case.expected_action == "follow":
            selection_denominator += 1
        decision_denominator += 1

    return (
        scheduled,
        valid,
        errors,
        missing,
        selection_numerator,
        selection_denominator,
        decision_numerator,
        decision_denominator,
        first_numerator,
        first_denominator,
    )


def _case_metrics(
    policy: PolicyName,
    trials: list[Trial],
    case_by_id: dict[str, GateCase],
    result_by_key: dict[tuple[str, str], ResultRecord],
    *,
    permutation_by_case: dict[str, MeanMetric],
    repeat_by_case: dict[str, MeanMetric],
) -> list[CaseMetrics]:
    by_case: dict[str, list[Trial]] = defaultdict(list)
    for trial in trials:
        by_case[trial.case_id].append(trial)

    metrics: list[CaseMetrics] = []
    for case_id in sorted(by_case):
        case = case_by_id[case_id]
        if case.expected_action not in {"follow", "stop"}:
            raise ValueError(f"scheduled case is not semantically labelled: {case_id}")
        (
            scheduled,
            valid,
            errors,
            missing,
            selection_numerator,
            selection_denominator,
            decision_numerator,
            decision_denominator,
            first_numerator,
            first_denominator,
        ) = _case_counters(case, by_case[case_id], result_by_key, policy)
        metrics.append(
            CaseMetrics(
                case_id=case_id,
                source_hop=case.source_hop,
                scheduled=scheduled,
                valid=valid,
                errors=errors,
                missing=missing,
                selection_accuracy=ratio_metric(selection_numerator, selection_denominator),
                decision_accuracy=ratio_metric(decision_numerator, decision_denominator),
                first_position_selection_rate=ratio_metric(first_numerator, first_denominator),
                permutation_consistency=permutation_by_case[case_id],
                repeat_consistency=repeat_by_case[case_id],
            )
        )
    return metrics


def _mean_from_ratios(ratios: Iterable[Any], eligible: int):
    values = [ratio.value for ratio in ratios if ratio.value is not None]
    return mean_metric(values, eligible=eligible, excluded=eligible - len(values))


def _question_sort_key(key: QuestionKey) -> tuple[str, str, str]:
    return key.dataset_id, type(key.question_id).__name__, str(key.question_id)


def _question_macro(
    metrics: list[CaseMetrics],
    case_by_id: dict[str, GateCase],
) -> QuestionMacroSummary:
    by_question_hop: dict[tuple[QuestionKey, int], list[CaseMetrics]] = defaultdict(list)
    for item in metrics:
        case = case_by_id[item.case_id]
        key = QuestionKey(dataset_id=case.dataset_id, question_id=case.question_id)
        by_question_hop[(key, item.source_hop)].append(item)

    question_selection: dict[QuestionKey, list[float]] = defaultdict(list)
    question_decision: dict[QuestionKey, list[float]] = defaultdict(list)
    question_permutation: dict[QuestionKey, list[float]] = defaultdict(list)
    question_repeat: dict[QuestionKey, list[float]] = defaultdict(list)
    question_keys: set[QuestionKey] = set()
    hops: list[QuestionHopMetrics] = []
    ordered = sorted(
        by_question_hop.items(),
        key=lambda item: (
            *_question_sort_key(item[0][0]),
            item[0][1],
        ),
    )
    for (key, source_hop), hop_metrics in ordered:
        question_keys.add(key)
        selection_values = [
            item.selection_accuracy.value
            for item in hop_metrics
            if item.selection_accuracy.value is not None
        ]
        decision_values = [
            item.decision_accuracy.value
            for item in hop_metrics
            if item.decision_accuracy.value is not None
        ]
        permutation_values = [
            item.permutation_consistency.value
            for item in hop_metrics
            if item.permutation_consistency.value is not None
        ]
        repeat_values = [
            item.repeat_consistency.value
            for item in hop_metrics
            if item.repeat_consistency.value is not None
        ]
        question_selection[key].extend(selection_values)
        question_decision[key].extend(decision_values)
        question_permutation[key].extend(permutation_values)
        question_repeat[key].extend(repeat_values)
        hops.append(
            QuestionHopMetrics(
                question_key=key,
                source_hop=source_hop,
                scheduled_cases=len(hop_metrics),
                defined_cases=len(decision_values),
                selection_accuracy=mean_metric(
                    selection_values,
                    eligible=len(hop_metrics),
                    excluded=len(hop_metrics) - len(selection_values),
                ),
                decision_accuracy=mean_metric(
                    decision_values,
                    eligible=len(hop_metrics),
                    excluded=len(hop_metrics) - len(decision_values),
                ),
                permutation_consistency=mean_metric(
                    permutation_values,
                    eligible=len(hop_metrics),
                    excluded=len(hop_metrics) - len(permutation_values),
                ),
                repeat_consistency=mean_metric(
                    repeat_values,
                    eligible=len(hop_metrics),
                    excluded=len(hop_metrics) - len(repeat_values),
                ),
            )
        )

    question_count = len(question_keys)
    selection_question_values = [
        sum(values) / len(values)
        for values in question_selection.values()
        if values
    ]
    decision_question_values = [
        sum(values) / len(values)
        for values in question_decision.values()
        if values
    ]
    permutation_question_values = [
        sum(values) / len(values)
        for values in question_permutation.values()
        if values
    ]
    repeat_question_values = [
        sum(values) / len(values)
        for values in question_repeat.values()
        if values
    ]
    return QuestionMacroSummary(
        question_count=question_count,
        selection_accuracy=mean_metric(
            selection_question_values,
            eligible=question_count,
            excluded=question_count - len(selection_question_values),
        ),
        decision_accuracy=mean_metric(
            decision_question_values,
            eligible=question_count,
            excluded=question_count - len(decision_question_values),
        ),
        permutation_consistency=mean_metric(
            permutation_question_values,
            eligible=question_count,
            excluded=question_count - len(permutation_question_values),
        ),
        repeat_consistency=mean_metric(
            repeat_question_values,
            eligible=question_count,
            excluded=question_count - len(repeat_question_values),
        ),
        hops=hops,
    )


def _gold_position_groups(
    trials: list[Trial],
    case_by_id: dict[str, GateCase],
    result_by_key: dict[tuple[str, str], ResultRecord],
    policy: PolicyName,
) -> list[GoldPositionGroup]:
    single_gold: dict[tuple[int, int], list[tuple[GateCase, Trial]]] = defaultdict(list)
    multi_gold: dict[int, list[tuple[GateCase, Trial]]] = defaultdict(list)

    for trial in trials:
        case = case_by_id[trial.case_id]
        if case.expected_action != "follow" or len(case.candidates) < MIN_MULTI_CANDIDATES:
            continue
        if len(case.acceptable_dieu) == 1:
            gold = next(iter(case.acceptable_dieu))
            position = trial.candidate_order.index(gold) + 1
            single_gold[(len(case.candidates), position)].append((case, trial))
        elif len(case.acceptable_dieu) > 1:
            multi_gold[len(case.candidates)].append((case, trial))

    cells: list[tuple[str, int, list[int], list[tuple[GateCase, Trial]]]] = []
    cells.extend(
        ("single_gold", key[0], [key[1]], pairs)
        for key, pairs in single_gold.items()
    )
    cells.extend(
        ("multi_gold", key, [], pairs)
        for key, pairs in multi_gold.items()
    )
    cells.sort(key=lambda cell: (cell[0], cell[1], cell[2]))

    groups: list[GoldPositionGroup] = []
    for view, candidate_count, gold_positions, pairs in cells:
        per_case: dict[str, list[bool]] = defaultdict(list)
        correct = 0
        for case, trial in pairs:
            result = result_by_key.get((policy, trial.trial_id))
            matched = (
                result is not None
                and selection_correctness(case, result.outcome) is True
            )
            per_case[case.case_id].append(matched)
            correct += matched
        case_views = [
            GoldPositionCase(
                case_id=case_id,
                correct=sum(values),
                scheduled=len(values),
                accuracy=ratio_metric(sum(values), len(values)),
            )
            for case_id, values in sorted(per_case.items())
        ]
        groups.append(
            GoldPositionGroup(
                view=view,
                candidate_count=candidate_count,
                gold_positions=gold_positions,
                correct=correct,
                scheduled=len(pairs),
                accuracy=ratio_metric(correct, len(pairs)),
                case_ids=[item.case_id for item in case_views],
                cases=case_views,
                case_macro=mean_metric(
                    [item.accuracy.value for item in case_views],
                    eligible=len(case_views),
                    excluded=0,
                ),
            )
        )
    return groups


def _gold_position_cell(
    case: GateCase,
    trial: Trial,
) -> tuple[str, int, int] | None:
    if case.expected_action != "follow" or len(case.acceptable_dieu) != 1:
        return None
    gold = next(iter(case.acceptable_dieu))
    return case.case_id, len(case.candidates), trial.candidate_order.index(gold) + 1


def _position_cohorts(
    trials: list[Trial],
    case_by_id: dict[str, GateCase],
) -> list[PositionCohort]:
    exposures: dict[tuple[int, int], set[str]] = defaultdict(set)
    scheduled: dict[tuple[str, int, int], int] = defaultdict(int)
    candidate_counts: set[int] = set()

    for trial in trials:
        candidate_count = len(trial.candidate_order)
        if candidate_count < MIN_MULTI_CANDIDATES:
            continue
        cell = _gold_position_cell(case_by_id[trial.case_id], trial)
        if cell is None:
            continue
        case_id, count, position = cell
        candidate_counts.add(count)
        exposures[(count, position)].add(case_id)
        scheduled[(case_id, count, position)] += 1

    cohorts: list[PositionCohort] = []
    for candidate_count in sorted(candidate_counts):
        positions = list(range(1, candidate_count + 1))
        case_sets = [
            exposures.get((candidate_count, position), set())
            for position in positions
        ]
        cohort_ids = set.intersection(*case_sets) if case_sets else set()
        if not cohort_ids:
            continue
        per_position = [
            sum(
                scheduled[(case_id, candidate_count, position)]
                for case_id in cohort_ids
            )
            for position in positions
        ]
        cohorts.append(
            PositionCohort(
                candidate_count=candidate_count,
                positions=positions,
                case_ids=sorted(cohort_ids),
                scheduled_per_position=min(per_position),
            )
        )
    return cohorts


def _position_distribution(
    condition: PermutationCondition,
    candidate_count: int | None,
    trials: list[Trial],
    result_by_key: dict[tuple[str, str], ResultRecord],
    policy: PolicyName,
) -> PositionDistribution:
    valid_results = [
        result_by_key[(policy, trial.trial_id)]
        for trial in trials
        if (policy, trial.trial_id) in result_by_key
        and result_by_key[(policy, trial.trial_id)].outcome.status == "ok"
    ]
    counts: defaultdict[str, int] = defaultdict(int)
    for result in valid_results:
        counts["stop" if result.selected_position is None else str(result.selected_position)] += 1
    max_position = max((len(trial.candidate_order) for trial in trials), default=0)
    return PositionDistribution(
        condition=condition,
        candidate_count=candidate_count,
        scheduled=len(trials),
        valid=len(valid_results),
        coverage=ratio_metric(len(valid_results), len(trials)),
        positions={
            str(position): ratio_metric(counts[str(position)], len(valid_results))
            for position in range(1, max_position + 1)
        },
        stop=ratio_metric(counts["stop"], len(valid_results)),
    )


def _position_distributions(
    condition: PermutationCondition,
    trials: list[Trial],
    result_by_key: dict[tuple[str, str], ResultRecord],
    policy: PolicyName,
) -> list[PositionDistribution]:
    multi = [
        trial for trial in trials
        if len(trial.candidate_order) >= MIN_MULTI_CANDIDATES
    ]
    by_count: dict[int, list[Trial]] = defaultdict(list)
    for trial in multi:
        by_count[len(trial.candidate_order)].append(trial)
    distributions = [
        _position_distribution(condition, None, multi, result_by_key, policy)
    ]
    distributions.extend(
        _position_distribution(condition, count, by_count[count], result_by_key, policy)
        for count in sorted(by_count)
    )
    return distributions


def _identical_inputs(
    trials: list[Trial],
    condition: PermutationCondition,
) -> list[IdenticalInputGroup]:
    if condition not in SEED_ORDER_CONDITIONS:
        return []
    by_input: dict[tuple[str, str, tuple[int, ...]], set[str]] = defaultdict(set)
    for trial in trials:
        key = (trial.case_id, trial.observation_hash, tuple(trial.candidate_order))
        by_input[key].add(trial.permutation_id)
    return [
        IdenticalInputGroup(
            case_id=case_id,
            condition=condition,
            observation_hash=observation_hash,
            candidate_order=list(candidate_order),
            permutation_ids=sorted(permutation_ids),
        )
        for (case_id, observation_hash, candidate_order), permutation_ids in sorted(
            by_input.items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2]),
        )
        if len(permutation_ids) > 1
    ]


def _condition_summary(
    policy: PolicyName,
    condition: PermutationCondition,
    trials: list[Trial],
    case_by_id: dict[str, GateCase],
    result_by_key: dict[tuple[str, str], ResultRecord],
) -> ConditionSummary:
    permutation_trials = list(trials)
    permutation_groups: dict[tuple[str, Any], list[Trial]] = defaultdict(list)
    repeat_groups: dict[tuple[str, Any], list[Trial]] = defaultdict(list)
    for trial in permutation_trials:
        permutation_groups[(trial.case_id, trial.repeat_id)].append(trial)
    for trial in trials:
        repeat_groups[(trial.case_id, trial.permutation_id)].append(trial)

    permutation_consistency = _consistency_summary(
        permutation_groups,
        result_by_key,
        policy,
        scheduled_cases={trial.case_id for trial in permutation_trials},
        min_candidate_count=(
            MIN_MULTI_CANDIDATES
            if condition in {"candidate-order", "combined"}
            else None
        ),
    )
    repeat_consistency = _consistency_summary(
        repeat_groups,
        result_by_key,
        policy,
        scheduled_cases={trial.case_id for trial in trials},
    )
    metrics = _case_metrics(
        policy,
        trials,
        case_by_id,
        result_by_key,
        permutation_by_case=permutation_consistency.by_case,
        repeat_by_case=repeat_consistency.by_case,
    )
    scheduled = sum(item.scheduled for item in metrics)
    valid = sum(item.valid for item in metrics)
    errors = sum(item.errors for item in metrics)
    missing = sum(item.missing for item in metrics)
    selection_numerator = sum(item.selection_accuracy.numerator for item in metrics)
    selection_denominator = sum(item.selection_accuracy.denominator for item in metrics)
    decision_numerator = sum(item.decision_accuracy.numerator for item in metrics)
    decision_denominator = sum(item.decision_accuracy.denominator for item in metrics)
    first_numerator = sum(item.first_position_selection_rate.numerator for item in metrics)
    first_denominator = sum(item.first_position_selection_rate.denominator for item in metrics)

    return ConditionSummary(
        scheduled=scheduled,
        valid=valid,
        errors=errors,
        missing=missing,
        selection_accuracy=ratio_metric(selection_numerator, selection_denominator),
        decision_accuracy=ratio_metric(decision_numerator, decision_denominator),
        first_position_selection_rate=ratio_metric(first_numerator, first_denominator),
        selection_case_macro=_mean_from_ratios(
            (item.selection_accuracy for item in metrics), len(metrics)
        ),
        decision_case_macro=_mean_from_ratios(
            (item.decision_accuracy for item in metrics), len(metrics)
        ),
        question_macro=_question_macro(metrics, case_by_id),
        cases=metrics,
        permutation_consistency=permutation_consistency.summary,
        repeat_consistency=repeat_consistency.summary,
        gold_positions=_gold_position_groups(trials, case_by_id, result_by_key, policy),
        position_cohorts=_position_cohorts(trials, case_by_id),
        position_distributions=_position_distributions(
            condition, trials, result_by_key, policy
        ),
        identical_inputs=_identical_inputs(trials, condition),
    )


def _paired_condition_summary(
    condition: PermutationCondition,
    policy_a: PolicyName,
    policy_b: PolicyName,
    trials: list[Trial],
    case_by_id: dict[str, GateCase],
    result_by_key: dict[tuple[str, str], ResultRecord],
) -> PairedConditionSummary:
    scheduled_pairs = valid_pairs = agreement = a_wins = b_wins = ties = 0
    for trial in trials:
        scheduled_pairs += 1
        first = result_by_key.get((policy_a, trial.trial_id))
        second = result_by_key.get((policy_b, trial.trial_id))
        if first is None or second is None:
            continue
        if first.outcome.status != "ok" or second.outcome.status != "ok":
            continue
        valid_pairs += 1
        if action_key_from_outcome(first.outcome) == action_key_from_outcome(second.outcome):
            agreement += 1
        case = case_by_id[trial.case_id]
        first_correct = decision_correctness(case, first.outcome)
        second_correct = decision_correctness(case, second.outcome)
        if first_correct and not second_correct:
            a_wins += 1
        elif second_correct and not first_correct:
            b_wins += 1
        else:
            ties += 1
    return PairedConditionSummary(
        condition=condition,
        scheduled_pairs=scheduled_pairs,
        valid_pairs=valid_pairs,
        coverage=ratio_metric(valid_pairs, scheduled_pairs),
        agreement=ratio_metric(agreement, valid_pairs),
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
    )


def _paired_policies(
    manifest: RunManifest,
    case_by_id: dict[str, GateCase],
    result_by_key: dict[tuple[str, str], ResultRecord],
) -> list[PairedPolicySummary]:
    conditions: list[PermutationCondition] = []
    for trial in manifest.trials:
        if trial.condition not in conditions:
            conditions.append(trial.condition)

    summaries: list[PairedPolicySummary] = []
    for policy_a, policy_b in combinations(manifest.policies, 2):
        by_condition: list[PairedConditionSummary] = []
        scheduled_pairs = valid_pairs = agreement = a_wins = b_wins = ties = 0
        for condition in conditions:
            condition_summary = _paired_condition_summary(
                condition,
                policy_a,
                policy_b,
                [trial for trial in manifest.trials if trial.condition == condition],
                case_by_id,
                result_by_key,
            )
            by_condition.append(condition_summary)
            scheduled_pairs += condition_summary.scheduled_pairs
            valid_pairs += condition_summary.valid_pairs
            agreement += condition_summary.agreement.numerator
            a_wins += condition_summary.a_wins
            b_wins += condition_summary.b_wins
            ties += condition_summary.ties
        summaries.append(
            PairedPolicySummary(
                policies=[policy_a, policy_b],
                scheduled_pairs=scheduled_pairs,
                valid_pairs=valid_pairs,
                coverage=ratio_metric(valid_pairs, scheduled_pairs),
                agreement=ratio_metric(agreement, valid_pairs),
                a_wins=a_wins,
                b_wins=b_wins,
                ties=ties,
                by_condition=by_condition,
            )
        )
    return summaries


def summarize_permutation(
    manifest: RunManifest,
    cases: list[GateCase],
    results: list[ResultRecord],
) -> PermutationSummary:
    """Tính lại mọi aggregate từ schedule và result đã được validate."""
    if manifest.experiment != "permutation" or manifest.permutation is None:
        raise ValueError("summarize_permutation requires a permutation manifest")
    case_by_id, _, result_by_key = validate_result_set(manifest, cases, results)

    conditions: list[PermutationCondition] = []
    for trial in manifest.trials:
        if trial.condition not in conditions:
            conditions.append(trial.condition)

    by_policy: dict[PolicyName, PolicyPermutationSummary] = {}
    for policy in manifest.policies:
        by_condition = {
            condition: _condition_summary(
                policy,
                condition,
                [trial for trial in manifest.trials if trial.condition == condition],
                case_by_id,
                result_by_key,
            )
            for condition in conditions
        }
        policy_results = [result for result in results if result.policy == policy]
        by_policy[policy] = PolicyPermutationSummary(
            scheduled=sum(item.scheduled for item in by_condition.values()),
            valid=sum(item.valid for item in by_condition.values()),
            errors=sum(item.errors for item in by_condition.values()),
            missing=sum(item.missing for item in by_condition.values()),
            telemetry=telemetry_summary(policy_results),
            by_condition=by_condition,
        )

    config = manifest.permutation
    return PermutationSummary(
        schema_version=manifest.schema_version,
        experiment="permutation",
        run_id=manifest.run_id,
        by_policy=by_policy,
        paired=_paired_policies(manifest, case_by_id, result_by_key),
        exclusions=list(manifest.exclusions),
        schedule={
            "condition": config.condition,
            "schedule": config.schedule,
            "repeats": config.repeats,
            "policies": list(manifest.policies),
            "scheduled_trials": len(manifest.trials),
            "expected_trial_count": manifest.expected_trial_count,
            "expected_policy_result_count": manifest.expected_policy_result_count,
            "persisted_results": len(results),
        },
    )


__all__ = ["summarize_permutation"]
