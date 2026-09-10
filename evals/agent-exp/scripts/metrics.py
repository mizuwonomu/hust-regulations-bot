"""Pure scoring and aggregation for initial-gate replay artifacts."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

try:
    from contracts import (
        DecisionOutcome,
        GateCase,
        PairedSummary,
        PolicyName,
        PolicySummary,
        RatioMetric,
        ResultRecord,
        RunManifest,
        Summary,
        Trial,
    )
except ModuleNotFoundError:
    from .contracts import (
        DecisionOutcome,
        GateCase,
        PairedSummary,
        PolicyName,
        PolicySummary,
        RatioMetric,
        ResultRecord,
        RunManifest,
        Summary,
        Trial,
    )


def _ratio(numerator: int, denominator: int) -> RatioMetric:
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        value=(numerator / denominator if denominator else None),
    )


def _decision_key(record: ResultRecord) -> tuple[bool, int | None]:
    decision = record.outcome.decision
    if decision is None:
        raise ValueError("a valid result must contain a decision")
    return decision.stop, None if decision.stop else decision.dieu


def _validate_result_set(
    manifest: RunManifest,
    cases: list[GateCase],
    results: list[ResultRecord],
) -> tuple[dict[str, GateCase], dict[str, Trial], dict[tuple[str, str], ResultRecord]]:
    case_by_id = {case.case_id: case for case in cases}
    trial_by_id = {trial.trial_id: trial for trial in manifest.trials}
    if len(case_by_id) != len(cases):
        raise ValueError("cases contain duplicate case_id values")
    if len(trial_by_id) != len(manifest.trials):
        raise ValueError("manifest contains duplicate trial_id values")

    for trial in manifest.trials:
        case = case_by_id.get(trial.case_id)
        if case is None:
            raise ValueError(f"trial uses an unknown case: {trial.case_id}")
        if trial.observation_hash != case.observation_hash:
            raise ValueError(f"trial changes the observation hash: {trial.trial_id}")
        if trial.candidate_order != case.candidates:
            raise ValueError(f"trial changes candidate order: {trial.trial_id}")

    result_by_key: dict[tuple[str, str], ResultRecord] = {}
    for result in results:
        if result.policy not in manifest.policies:
            raise ValueError(f"result uses an unscheduled policy: {result.policy}")
        trial = trial_by_id.get(result.trial.trial_id)
        if trial is None:
            raise ValueError(f"result uses an unknown trial: {result.trial.trial_id}")
        if result.trial.model_dump(mode="json") != trial.model_dump(mode="json"):
            raise ValueError(f"result changes the saved trial: {result.trial.trial_id}")
        case = case_by_id.get(trial.case_id)
        if case is None:
            raise ValueError(f"trial uses an unknown case: {trial.case_id}")
        if result.run_id != manifest.run_id:
            raise ValueError(f"result uses a different run_id: {result.trial.trial_id}")
        if result.expected_action != case.expected_action:
            raise ValueError(f"result label differs from case: {result.trial.trial_id}")
        if result.acceptable_dieu != case.acceptable_dieu:
            raise ValueError(f"result acceptable_dieu differs from case: {result.trial.trial_id}")
        key = (result.policy, result.trial.trial_id)
        if key in result_by_key:
            raise ValueError(f"duplicate result key: {key}")
        result_by_key[key] = result
    return case_by_id, trial_by_id, result_by_key


def score_result(
    run_id: str,
    policy: PolicyName,
    trial: Trial,
    case: GateCase,
    outcome: DecisionOutcome,
) -> ResultRecord:
    """Score one policy outcome against an approved gate label."""
    if case.expected_action not in {"follow", "stop"}:
        raise ValueError("only approved follow and stop cases can be scored")
    if trial.case_id != case.case_id:
        raise ValueError("trial and case identifiers do not match")
    if trial.observation_hash != case.observation_hash:
        raise ValueError("trial observation hash does not match case")
    if trial.candidate_order != case.candidates:
        raise ValueError("trial candidate order does not match case")

    selected_position: int | None = None
    selection_correct: bool | None = None
    decision_correct = False

    if outcome.status == "error":
        selection_correct = False if case.expected_action == "follow" else None
    else:
        if outcome.decision is None:
            raise ValueError("ok outcome is missing a decision")
        decision = outcome.decision
        if decision.stop:
            if case.expected_action == "stop":
                decision_correct = True
            else:
                selection_correct = False
        else:
            if decision.dieu is None or decision.dieu not in trial.candidate_order:
                raise ValueError("follow decision selected an article outside candidates")
            selected_position = trial.candidate_order.index(decision.dieu) + 1
            if case.expected_action == "follow":
                selection_correct = decision.dieu in case.acceptable_dieu
                decision_correct = selection_correct

    return ResultRecord(
        run_id=run_id,
        policy=policy,
        trial=trial,
        source_hop=case.source_hop,
        expected_action=case.expected_action,
        acceptable_dieu=set(case.acceptable_dieu),
        outcome=outcome,
        selected_position=selected_position,
        selection_correct=selection_correct,
        decision_correct=decision_correct,
    )


def _telemetry(records: list[ResultRecord]) -> dict[str, Any]:
    latencies = [record.outcome.latency_ms for record in records]
    latency: dict[str, Any] = {
        "count": len(latencies),
        "mean": (sum(latencies) / len(latencies) if latencies else None),
        "min": (min(latencies) if latencies else None),
        "max": (max(latencies) if latencies else None),
    }
    for key in ("mean", "min", "max"):
        if isinstance(latency[key], float) and not math.isfinite(latency[key]):
            latency[key] = None

    usage: dict[str, dict[str, int | None]] = {}
    for field_name in ("input_tokens", "output_tokens", "total_tokens"):
        values = [
            getattr(record.outcome.usage, field_name)
            for record in records
            if record.outcome.usage is not None
            and getattr(record.outcome.usage, field_name) is not None
        ]
        usage[field_name] = {
            "count": len(values),
            "sum": sum(values) if values else None,
        }
    return {"latency_ms": latency, "usage": usage}


def _summarize_policy(
    trials: list[Trial],
    cases: dict[str, GateCase],
    result_by_key: dict[tuple[str, str], ResultRecord],
    policy: PolicyName,
) -> PolicySummary:
    scheduled = len(trials)
    valid = 0
    errors = 0
    missing = 0
    selection_numerator = 0
    selection_denominator = 0
    decision_numerator = 0
    decision_denominator = 0
    first_position_numerator = 0
    first_position_denominator = 0
    records: list[ResultRecord] = []

    for trial in trials:
        case = cases[trial.case_id]
        if case.expected_action not in {"follow", "stop"}:
            raise ValueError(f"unscheduled semantic label in trial {trial.trial_id}")
        result = result_by_key.get((policy, trial.trial_id))
        if result is None:
            missing += 1
            if case.expected_action == "follow":
                selection_denominator += 1
            decision_denominator += 1
            continue

        records.append(result)
        if result.outcome.status == "error":
            errors += 1
        else:
            valid += 1
        if case.expected_action == "follow":
            selection_denominator += 1
            if result.selection_correct is True:
                selection_numerator += 1
        decision_denominator += 1
        if result.decision_correct:
            decision_numerator += 1
        if len(trial.candidate_order) >= 2 and result.outcome.status == "ok":
            first_position_denominator += 1
            if result.selected_position == 1:
                first_position_numerator += 1

    return PolicySummary(
        scheduled=scheduled,
        valid=valid,
        errors=errors,
        missing=missing,
        selection_accuracy=_ratio(selection_numerator, selection_denominator),
        decision_accuracy=_ratio(decision_numerator, decision_denominator),
        first_position_selection_rate=_ratio(
            first_position_numerator,
            first_position_denominator,
        ),
        costs=_telemetry(records),
    )


def _position_metrics(
    trials: list[Trial],
    result_by_key: dict[tuple[str, str], ResultRecord],
    policy: PolicyName,
) -> dict[str, RatioMetric]:
    eligible = [
        result_by_key[(policy, trial.trial_id)]
        for trial in trials
        if (policy, trial.trial_id) in result_by_key
        and result_by_key[(policy, trial.trial_id)].outcome.status == "ok"
        and len(trial.candidate_order) >= 2
    ]
    denominator = len(eligible)
    counts: defaultdict[str, int] = defaultdict(int)
    for result in eligible:
        key = "stop" if result.selected_position is None else str(result.selected_position)
        counts[key] += 1
    max_position = max(
        (len(trial.candidate_order) for trial in trials if len(trial.candidate_order) >= 2),
        default=0,
    )
    keys = [str(position) for position in range(1, max_position + 1)]
    keys.append("stop")
    return {key: _ratio(counts[key], denominator) for key in keys}


def _paired_summary(
    trials: list[Trial],
    result_by_key: dict[tuple[str, str], ResultRecord],
) -> PairedSummary:
    valid_pairs = 0
    incomplete_pairs = 0
    agreement_numerator = 0
    first_wins = 0
    llm_wins = 0
    ties = 0
    different_trial_ids: list[str] = []

    for trial in trials:
        first = result_by_key.get(("first", trial.trial_id))
        llm = result_by_key.get(("llm", trial.trial_id))
        if first is None or llm is None or first.outcome.status != "ok" or llm.outcome.status != "ok":
            incomplete_pairs += 1
            continue
        valid_pairs += 1
        if _decision_key(first) == _decision_key(llm):
            agreement_numerator += 1
        else:
            different_trial_ids.append(trial.trial_id)
        if first.decision_correct and not llm.decision_correct:
            first_wins += 1
        elif llm.decision_correct and not first.decision_correct:
            llm_wins += 1
        else:
            ties += 1

    return PairedSummary(
        scheduled_pairs=len(trials),
        valid_pairs=valid_pairs,
        incomplete_pairs=incomplete_pairs,
        agreement=_ratio(agreement_numerator, valid_pairs),
        first_wins=first_wins,
        llm_wins=llm_wins,
        ties=ties,
        different_trial_ids=different_trial_ids,
    )


def summarize(
    manifest: RunManifest,
    cases: list[GateCase],
    results: list[ResultRecord],
) -> Summary:
    """Recompute all initial-selection metrics from the saved schedule and rows."""
    case_by_id, trial_by_id, result_by_key = _validate_result_set(manifest, cases, results)
    trials = [trial_by_id[trial.trial_id] for trial in manifest.trials]
    by_policy = {
        policy: _summarize_policy(trials, case_by_id, result_by_key, policy)
        for policy in manifest.policies
    }

    group_trials: dict[str, list[Trial]] = {
        "single_candidate": [],
        "multi_candidate": [],
    }
    for trial in trials:
        group = "multi_candidate" if len(trial.candidate_order) >= 2 else "single_candidate"
        group_trials[group].append(trial)
    by_candidate_group = {
        group: {
            policy: _summarize_policy(group_trials[group], case_by_id, result_by_key, policy)
            for policy in manifest.policies
        }
        for group in group_trials
    }
    position_metrics = {
        policy: _position_metrics(trials, result_by_key, policy)
        for policy in manifest.policies
    }
    position_keys = sorted(
        {
            position
            for metrics in position_metrics.values()
            for position in metrics
        },
        key=lambda value: (value == "stop", int(value) if value != "stop" else 0),
    )
    by_position = {
        position: {
            policy: position_metrics[policy][position]
            for policy in manifest.policies
        }
        for position in position_keys
    }

    paired = None
    if "first" in manifest.policies and "llm" in manifest.policies:
        paired = _paired_summary(trials, result_by_key)

    return Summary(
        by_policy=by_policy,
        by_candidate_group=by_candidate_group,
        by_position=by_position,
        paired=paired,
        exclusions=list(manifest.exclusions),
    )


__all__ = ["score_result", "summarize"]
