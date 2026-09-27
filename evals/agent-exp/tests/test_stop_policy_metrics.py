"""Kiểm tra metric STOP bằng fixture tính tay, không dùng corpus hay model"""

from __future__ import annotations

import json

import pytest

from contracts import (
    DecisionOutcome,
    SourceFile,
    TrialError,
)
from src.rag.agent.schema import Decision

from diagnostic_subexp.shared.contracts import sha256_text, structured_hash

from stop_policy_eval.contracts import (
    BASELINE_VARIANT,
    REQUIREMENT_CLASSES,
    STOP_POLICY_SCHEMA_VERSION,
    StopExclusion,
    StopPolicyCase,
    StopReadiness,
    StopRequirementCount,
    StopResult,
    StopRunManifest,
    StopTrial,
)
from stop_policy_eval.metrics import derive_result_score, summarize_stop_run

OBSERVATION = "Các Điều đã thu thập:\n[Context 10] Điều 10. Ví dụ\n- Xem Điều 20 Quy chế này."


def make_case(
    case_id: str,
    *,
    expected_action: str,
    candidates: list[int],
    acceptable_dieu: list[int] | None = None,
    semantic_group_id: str | None = None,
    split: str = "dev",
    source_hop: int = 0,
) -> StopPolicyCase:
    """Dựng case đã duyệt tối thiểu cho fixture metric"""
    observation = f"{OBSERVATION}\n{case_id}"
    payload = {
        "case_id": case_id,
        "dataset_id": "metrics",
        "question_id": case_id,
        "snapshot_id": "seeds-metrics",
        "snapshot_hash": "a" * 64,
        "question": f"question {case_id}",
        "observation": observation,
        "observation_hash": sha256_text(observation),
        "candidates": list(candidates),
        "source_hop": source_hop,
        "source_run_id": None,
        "source_policy": None,
        "source_trial_id": None,
        "parent_case_id": None,
        "followed_dieu": None,
        "source_state_hash": None,
        "expected_action": expected_action,
        "acceptable_dieu": list(acceptable_dieu or []),
        "label_reason": "fixture rationale",
        "label_status": "approved",
        "label_observation_hash": sha256_text(observation),
        "semantic_group_id": semantic_group_id or f"group-{case_id}",
        "split": split,
        "fewshot_overlap": False,
        "source_row_hash": "b" * 64,
        "source_dataset_hash": "c" * 64,
    }
    if source_hop > 0:
        article_hash = "d" * 64
        collected_article_refs = [
            {"dieu": 1000 + index, "sha256": f"{index + 1:064x}"}
            for index in range(source_hop - 1)
        ]
        collected_article_refs.append({"dieu": candidates[0], "sha256": article_hash})
        state = {
            "dataset_id": payload["dataset_id"],
            "question_id": payload["question_id"],
            "source_hop": source_hop,
            "question": payload["question"],
            "observation": observation,
            "candidates": list(candidates),
            "source_run_id": "run-fixture",
            "source_policy": "llm",
            "source_trial_id": f"{case_id}:r0",
            "parent_case_id": f"parent-{case_id}",
            "followed_dieu": candidates[0],
            "followed_article_sha256": article_hash,
            "collected_article_refs": collected_article_refs,
        }
        payload.update(
            {
                "source_run_id": state["source_run_id"],
                "source_policy": state["source_policy"],
                "source_trial_id": state["source_trial_id"],
                "parent_case_id": state["parent_case_id"],
                "followed_dieu": state["followed_dieu"],
                "followed_article_sha256": state["followed_article_sha256"],
                "collected_article_refs": state["collected_article_refs"],
                "source_state_hash": structured_hash(state),
            }
        )
    return StopPolicyCase.model_validate(payload)


def make_trial(case: StopPolicyCase, repeat_id: int = 0) -> StopTrial:
    """Dựng trial khớp một case fixture"""
    return StopTrial(
        trial_id=f"{case.case_id}:r{repeat_id}",
        input_trace_id=f"{case.case_id}:{BASELINE_VARIANT}",
        case_id=case.case_id,
        variant=BASELINE_VARIANT,
        repeat_id=repeat_id,
        input_fingerprint=sha256_text(f"{case.case_id}:r{repeat_id}"),
    )


def make_manifest(
    *,
    policies: list[str],
    trials: list[StopTrial],
    split: str = "dev",
    repeats: int = 1,
    exclusions: list[StopExclusion] | None = None,
) -> StopRunManifest:
    """Dựng manifest đã validate tối thiểu cho fixture metric"""
    requirements = [
        StopRequirementCount(requirement=name, eligible_cases=1, satisfied=True)
        for name in REQUIREMENT_CLASSES
    ]
    return StopRunManifest(
        schema_version=STOP_POLICY_SCHEMA_VERSION,
        experiment="stop-policy",
        run_id="run-metrics",
        started_at="2026-01-01T00:00:00+00:00",
        status="complete",
        variant=BASELINE_VARIANT,
        split=split,
        policies=list(policies),
        repeats=repeats,
        cases_source=SourceFile(path="evals/agent-exp/cases.jsonl", sha256="a" * 64),
        snapshot_source=SourceFile(path="evals/agent-exp/seeds.json", sha256="b" * 64),
        corpus_sha256="c" * 64,
        exclusions=list(exclusions or []),
        trials=list(trials),
        input_trace_ids=sorted({trial.input_trace_id for trial in trials}),
        readiness=StopReadiness(split=split, requirements=requirements, complete=True),
        execution_config=None,
        execution_config_reason="metrics fixture has no client",
        policy_config={policy: {"status": "fixture"} for policy in policies},
        expected_trial_count=len(trials),
        expected_policy_result_count=len(trials) * len(policies),
        execution_revision="d" * 40,
        execution_dirty=False,
        executed_module_hashes={"stop_policy_eval/contracts.py": "e" * 64},
    )


def follow_outcome(dieu: int) -> DecisionOutcome:
    return DecisionOutcome(
        status="ok",
        decision=Decision(stop=False, dieu=dieu),
        error=None,
        latency_ms=1.0,
        usage=None,
    )


def stop_outcome() -> DecisionOutcome:
    return DecisionOutcome(
        status="ok",
        decision=Decision(stop=True, dieu=None),
        error=None,
        latency_ms=1.0,
        usage=None,
    )


def error_outcome(category: str = "timeout") -> DecisionOutcome:
    return DecisionOutcome(
        status="error",
        decision=None,
        error=TrialError(category=category, message="fixture failure"),
        latency_ms=2.0,
        usage=None,
    )


def make_result(case: StopPolicyCase, trial: StopTrial, policy: str, outcome: DecisionOutcome) -> StopResult:
    """Dựng result fixture; cờ đã lưu được tính lại từ chính outcome"""
    score = derive_result_score(case, list(case.candidates), outcome)
    return StopResult(
        run_id="run-metrics",
        trial_id=trial.trial_id,
        policy=policy,
        input_trace_id=trial.input_trace_id,
        request_fingerprint=sha256_text(f"{policy}:{trial.trial_id}"),
        outcome=outcome,
        **score,
    )


def summarize(policies, cases, trials, results, *, split="dev"):
    manifest = make_manifest(policies=policies, trials=trials, split=split)
    return summarize_stop_run(manifest, cases, results)


def test_always_follow_fixture_has_full_follow_and_over_hop_rates():
    follow_case = make_case("follow-1", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    stop_case = make_case("stop-1", expected_action="stop", candidates=[30])
    trials = [make_trial(follow_case), make_trial(stop_case)]
    results = [
        make_result(follow_case, trials[0], "llm", follow_outcome(10)),
        make_result(stop_case, trials[1], "llm", follow_outcome(30)),
    ]
    summary = summarize(["llm"], [follow_case, stop_case], trials, results)
    rates = summary.by_policy["llm"].label_rates
    assert rates.follow_accuracy.value == 1.0
    assert rates.false_stop_rate.value == 0.0
    assert rates.stop_accuracy.value == 0.0
    assert rates.over_hop_rate.value == 1.0
    assert rates.follow_valid == 1 and rates.stop_valid == 1
    policy = summary.by_policy["llm"]
    # FOLLOW luôn đúng action ở case follow, nhưng là over-hop ở case stop
    assert policy.action_accuracy.value == 0.5
    assert policy.selection_accuracy.value == 0.0
    assert policy.decision_accuracy.value == 0.0
    assert policy.scheduled == 2 and policy.valid == 2
    assert policy.errors == 0 and policy.missing == 0


def test_always_stop_fixture_has_the_opposite_rates():
    follow_case = make_case("follow-2", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    stop_case = make_case("stop-2", expected_action="stop", candidates=[30])
    trials = [make_trial(follow_case), make_trial(stop_case)]
    results = [
        make_result(follow_case, trials[0], "llm", stop_outcome()),
        make_result(stop_case, trials[1], "llm", stop_outcome()),
    ]
    summary = summarize(["llm"], [follow_case, stop_case], trials, results)
    rates = summary.by_policy["llm"].label_rates
    assert rates.follow_accuracy.value == 0.0
    assert rates.false_stop_rate.value == 1.0
    assert rates.stop_accuracy.value == 1.0
    assert rates.over_hop_rate.value == 0.0
    assert summary.by_policy["llm"].action_accuracy.value == 0.5
    assert summary.by_policy["llm"].selection_accuracy.value == 0.0
    assert summary.by_policy["llm"].decision_accuracy.value == 0.5


def test_wrong_article_follow_is_action_correct_but_decision_incorrect():
    case = make_case("follow-3", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    trial = make_trial(case)
    summary = summarize(
        ["llm"], [case], [trial], [make_result(case, trial, "llm", follow_outcome(10))]
    )
    policy = summary.by_policy["llm"]
    assert policy.action_accuracy.value == 1.0
    assert policy.selection_accuracy.value == 0.0
    assert policy.decision_accuracy.value == 0.0
    assert policy.label_rates.follow_accuracy.value == 1.0
    assert policy.label_rates.false_stop_rate.value == 0.0
    assert policy.label_rates.stop_accuracy.value is None


def test_correct_follow_and_correct_stop_keep_selection_null_for_stop():
    follow_case = make_case("follow-4", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    stop_case = make_case("stop-4", expected_action="stop", candidates=[30])
    trials = [make_trial(follow_case), make_trial(stop_case)]
    results = [
        make_result(follow_case, trials[0], "llm", follow_outcome(20)),
        make_result(stop_case, trials[1], "llm", stop_outcome()),
    ]
    summary = summarize(["llm"], [follow_case, stop_case], trials, results)
    policy = summary.by_policy["llm"]
    assert policy.action_accuracy.value == 1.0
    assert policy.selection_accuracy.value == 1.0
    assert policy.decision_accuracy.value == 1.0
    macro = policy.case_macro
    assert macro.case_count == 2
    assert macro.selection_accuracy.defined == 1
    assert macro.selection_accuracy.excluded == 1
    assert macro.selection_accuracy.value == 1.0
    assert macro.follow_accuracy.defined == 1
    assert macro.stop_accuracy.defined == 1
    assert macro.action_accuracy.defined == 2
    assert macro.decision_accuracy.defined == 2


def test_error_and_missing_slots_stay_incorrect_and_are_reported_by_label():
    follow_case = make_case("follow-5", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    stop_case = make_case("stop-5", expected_action="stop", candidates=[30])
    trials = [make_trial(follow_case), make_trial(stop_case)]
    results = [make_result(follow_case, trials[0], "llm", error_outcome("timeout"))]
    summary = summarize(["llm"], [follow_case, stop_case], trials, results)
    policy = summary.by_policy["llm"]
    assert policy.scheduled == 2 and policy.valid == 0
    assert policy.errors == 1 and policy.missing == 1
    assert policy.action_accuracy.value == 0.0
    assert policy.selection_accuracy.value == 0.0
    assert policy.decision_accuracy.value == 0.0
    assert policy.label_rates.follow_accuracy.value is None
    assert policy.label_rates.stop_accuracy.value is None
    assert policy.label_rates.follow_valid == 0 and policy.label_rates.stop_valid == 0
    assert policy.error_rate["follow"].value == 1.0
    assert policy.error_rate["follow"].denominator == 1
    assert policy.error_rate["stop"].value == 0.0
    assert policy.missing_rate["follow"].value == 0.0
    assert policy.missing_rate["stop"].value == 1.0
    assert policy.case_macro.action_accuracy.defined == 2
    assert policy.case_macro.follow_accuracy.defined == 0
    assert policy.case_macro.follow_accuracy.excluded == 2
    assert summary.scheduled == 2 and summary.valid == 0
    assert summary.errors == 1 and summary.missing == 1


def test_complementary_action_rates_sum_to_one_when_defined():
    follow_case = make_case("follow-6", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    stop_case = make_case("stop-6", expected_action="stop", candidates=[30])
    trials = [make_trial(follow_case), make_trial(stop_case)]
    results = [
        make_result(follow_case, trials[0], "llm", follow_outcome(20)),
        make_result(stop_case, trials[1], "llm", stop_outcome()),
    ]
    rates = summarize(["llm"], [follow_case, stop_case], trials, results).by_policy["llm"].label_rates
    assert rates.follow_accuracy.value + rates.false_stop_rate.value == 1.0
    assert rates.stop_accuracy.value + rates.over_hop_rate.value == 1.0


def test_pooled_and_case_macro_rates_differ_for_unequal_case_sizes():
    small = make_case("macro-small", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    large = make_case("macro-large", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    trials = [make_trial(small, 0)] + [make_trial(large, repeat) for repeat in range(3)]
    results = [
        make_result(small, trials[0], "llm", follow_outcome(20)),
        make_result(large, trials[1], "llm", follow_outcome(20)),
        make_result(large, trials[2], "llm", follow_outcome(10)),
        make_result(large, trials[3], "llm", follow_outcome(10)),
    ]
    summary = summarize(["llm"], [small, large], trials, results)
    policy = summary.by_policy["llm"]
    assert policy.decision_accuracy.numerator == 2
    assert policy.decision_accuracy.denominator == 4
    assert policy.decision_accuracy.value == 0.5
    assert policy.case_macro.decision_accuracy.defined == 2
    assert policy.case_macro.decision_accuracy.value == pytest.approx((1.0 + 1 / 3) / 2)
    assert policy.case_macro.decision_accuracy.value != policy.decision_accuracy.value
    assert policy.scheduled == 4


def test_strata_retain_counts_per_case_group_hop_and_split():
    follow_case = make_case("strata-follow", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    hop_case = make_case(
        "strata-hop",
        expected_action="stop",
        candidates=[30],
        source_hop=1,
    )
    trials = [make_trial(follow_case), make_trial(hop_case)]
    results = [
        make_result(follow_case, trials[0], "llm", follow_outcome(10)),
        make_result(hop_case, trials[1], "llm", error_outcome("schema")),
    ]
    summary = summarize(["llm"], [follow_case, hop_case], trials, results)
    rows = {row.kind: {} for row in summary.strata}
    for row in summary.strata:
        rows[row.kind][row.value] = row
    assert set(rows) == {"case", "semantic_group", "source_hop", "split"}
    assert rows["case"]["strata-follow"].scheduled == 1
    assert rows["case"]["strata-follow"].valid == 1
    assert rows["case"]["strata-hop"].errors == 1
    assert rows["source_hop"]["hop0"].scheduled == 1
    assert rows["source_hop"]["post_follow"].errors == 1
    assert rows["split"]["dev"].scheduled == 2
    assert rows["semantic_group"]["group-strata-follow"].decision_accuracy.value == 0.0
    assert sorted(rows["case"]) == ["strata-follow", "strata-hop"]


def test_paired_comparison_separates_action_agreement_from_decision_wins():
    follow_case = make_case("pair-follow", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    stop_case = make_case("pair-stop", expected_action="stop", candidates=[30])
    trials = [make_trial(follow_case), make_trial(stop_case)]
    results = [
        make_result(follow_case, trials[0], "first", follow_outcome(10)),
        make_result(follow_case, trials[0], "llm", follow_outcome(20)),
        make_result(stop_case, trials[1], "first", follow_outcome(30)),
        make_result(stop_case, trials[1], "llm", stop_outcome()),
    ]
    summary = summarize(["first", "llm"], [follow_case, stop_case], trials, results)
    assert summary.paired is not None
    assert summary.paired.policies == ["first", "llm"]
    assert summary.paired.scheduled_pairs == 2
    assert summary.paired.valid_pairs == 2
    assert summary.paired.incomplete_pairs == 0
    assert summary.paired.action_agreement.value == 0.0
    assert summary.paired.a_wins == 0
    assert summary.paired.b_wins == 2
    assert summary.paired.ties == 0
    assert sorted(summary.paired.different_trial_ids) == sorted(trial.trial_id for trial in trials)
    assert summary.paired.policies == ["first", "llm"]


def test_stored_correctness_flags_do_not_drive_the_summary():
    case = make_case("stale-flags", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    trial = make_trial(case)
    result = make_result(case, trial, "llm", follow_outcome(20))
    stale = result.model_copy(update={"decision_correct": False, "selection_correct": False})
    summary = summarize(["llm"], [case], [trial], [stale])
    assert summary.by_policy["llm"].decision_accuracy.value == 1.0
    assert summary.by_policy["llm"].selection_accuracy.value == 1.0


def test_summary_recomputation_is_byte_stable():
    follow_case = make_case("stable-follow", expected_action="follow", candidates=[10, 20], acceptable_dieu=[20])
    stop_case = make_case("stable-stop", expected_action="stop", candidates=[30])
    trials = [make_trial(follow_case), make_trial(stop_case)]
    results = [
        make_result(follow_case, trials[0], "llm", follow_outcome(20)),
        make_result(stop_case, trials[1], "llm", stop_outcome()),
    ]
    manifest = make_manifest(policies=["llm"], trials=trials, exclusions=[StopExclusion(case_id="excluded", reason="draft")])
    first = summarize_stop_run(manifest, [follow_case, stop_case], results)
    second = summarize_stop_run(manifest, [follow_case, stop_case], results)
    assert json.dumps(first.model_dump(mode="json"), sort_keys=True) == json.dumps(
        second.model_dump(mode="json"), sort_keys=True
    )
    assert [item.case_id for item in first.exclusions] == ["excluded"]
    assert first.readiness.split == "dev"


def test_summary_rejects_a_duplicate_or_unscheduled_result():
    case = make_case("dup", expected_action="stop", candidates=[30])
    trial = make_trial(case)
    manifest = make_manifest(policies=["llm"], trials=[trial])
    result = make_result(case, trial, "llm", stop_outcome())
    with pytest.raises(ValueError, match="duplicate"):
        summarize_stop_run(manifest, [case], [result, result])
    other = make_result(case, trial, "first", stop_outcome())
    with pytest.raises(ValueError, match="policy"):
        summarize_stop_run(manifest, [case], [other])
