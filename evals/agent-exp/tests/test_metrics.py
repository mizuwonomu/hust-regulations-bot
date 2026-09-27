"""Tests metric: chấm điểm theo fixture tính tay, mẫu số 0 ra null, paired, nhóm candidate."""

from __future__ import annotations

import pytest

from contracts import (
    DecisionOutcome,
    GateCase,
    PermutationConfig,
    PERMUTATION_SCHEMA_VERSION,
    RunManifest,
    SourceFile,
    Trial,
    TrialError,
)
from metrics import score_result, summarize
from run_experiments import plan_trials

from src.rag.agent.schema import Decision
from synthetic import error_outcome, make_case, make_trial, ok_outcome


def _case(case_id: str, action: str, candidates: list[int], acceptable: list[int]):
    return GateCase(
        case_id=case_id,
        dataset_id="dataset",
        question_id=case_id,
        snapshot_id="snapshot",
        snapshot_hash="snapshot-hash",
        question=f"question {case_id}",
        observation=f"observation {case_id}",
        observation_hash=f"observation-hash-{case_id}",
        candidates=candidates,
        source_hop=0,
        source_run_id=None,
        source_policy=None,
        expected_action=action,
        acceptable_dieu=acceptable,
        label_reason="approved label",
        label_status="approved",
        split="dev",
        fewshot_overlap=None,
    )


def _manifest(trials, policies=("first", "llm")):
    return RunManifest(
        schema_version=1,
        run_id="run",
        experiment="initial-selection",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at=None,
        status="running",
        policies=list(policies),
        trials=trials,
        cases_source=SourceFile(path="cases.jsonl", sha256="cases-hash"),
        snapshot_source=SourceFile(path="snapshot.json", sha256="snapshot-file-hash"),
        exclusions=[],
        provenance={},
        policy_config={policy: {} for policy in policies},
    )


def _outcome(decision, latency=1.0):
    return DecisionOutcome(
        status="ok",
        decision=decision,
        error=None,
        latency_ms=latency,
        usage=None,
    )


def test_selection_decision_position_and_pair_metrics():
    # Fixture tính tay: candidate đúng ở vị trí 2 - first sai, LLM stub đúng
    cases = [
        _case("follow", "follow", [2, 3], [3]),
        _case("stop", "stop", [4], []),
    ]
    trials = plan_trials(cases, repeats=1)
    manifest = _manifest(trials)
    by_case = {case.case_id: case for case in cases}
    results = [
        score_result(
            "run",
            "first",
            trials[0],
            by_case[trials[0].case_id],
            _outcome(Decision(stop=False, dieu=2)),
        ),
        score_result(
            "run",
            "first",
            trials[1],
            by_case[trials[1].case_id],
            _outcome(Decision(stop=False, dieu=4)),
        ),
    ]
    results.extend(
        [
            score_result(
                "run",
                "llm",
                trials[0],
                by_case[trials[0].case_id],
                _outcome(Decision(stop=False, dieu=3), latency=2.0),
            ),
            score_result(
                "run",
                "llm",
                trials[1],
                by_case[trials[1].case_id],
                _outcome(Decision(stop=True, dieu=None), latency=2.0),
            ),
        ]
    )

    summary = summarize(manifest, cases, results)
    first_summary = summary.by_policy["first"]
    llm_summary = summary.by_policy["llm"]
    assert first_summary.selection_accuracy.model_dump() == {
        "numerator": 0,
        "denominator": 1,
        "value": 0.0,
    }
    assert first_summary.decision_accuracy.value == 0.0
    # First-position rate là per-policy: first có 1 valid output, chọn vị trí 1
    assert first_summary.first_position_selection_rate.value == 1.0
    assert llm_summary.selection_accuracy.value == 1.0
    assert llm_summary.decision_accuracy.value == 1.0
    assert llm_summary.first_position_selection_rate.value == 0.0
    assert summary.paired is not None
    assert summary.paired.agreement.value == 0.0
    assert summary.paired.first_wins == 0
    assert summary.paired.llm_wins == 2
    assert summary.paired.ties == 0
    assert summary.paired.different_trial_ids == [trials[0].trial_id, trials[1].trial_id]


def test_errors_and_missing_trials_stay_in_primary_denominators():
    cases = [
        _case("follow", "follow", [2, 3], [3]),
        _case("stop", "stop", [4, 5], []),
    ]
    trials = plan_trials(cases, repeats=2)
    manifest = _manifest(trials, policies=("first",))
    by_case = {case.case_id: case for case in cases}
    error = DecisionOutcome(
        status="error",
        decision=None,
        error=TrialError(category="timeout", message="timed out"),
        latency_ms=3.0,
        usage=None,
    )
    results = [
        score_result("run", "first", trials[0], by_case["follow"], error),
        score_result(
            "run",
            "first",
            trials[2],
            by_case["stop"],
            _outcome(Decision(stop=True, dieu=None)),
        ),
    ]
    summary = summarize(manifest, cases, results).by_policy["first"]
    assert summary.scheduled == 4
    assert summary.valid == 1
    assert summary.errors == 1
    assert summary.missing == 2
    assert summary.selection_accuracy.numerator == 0
    assert summary.selection_accuracy.denominator == 2
    assert summary.decision_accuracy.numerator == 1
    assert summary.decision_accuracy.denominator == 4
    assert summary.first_position_selection_rate.numerator == 0
    assert summary.first_position_selection_rate.denominator == 1


def test_stop_on_follow_label_is_scored_incorrect_and_counted_valid():
    # STOP trên case bắt follow: sai Selection Accuracy (False), không phải null
    case = _case("follow", "follow", [2, 3], [3])
    trial = plan_trials([case], repeats=1)[0]
    record = score_result(
        "run",
        "first",
        trial,
        case,
        _outcome(Decision(stop=True, dieu=None)),
    )

    assert record.selected_position is None
    assert record.selection_correct is False
    assert record.decision_correct is False
    summary = summarize(_manifest([trial], policies=("first",)), [case], [record]).by_policy["first"]
    assert summary.valid == 1
    assert summary.errors == 0
    assert summary.decision_accuracy.model_dump() == {
        "numerator": 0,
        "denominator": 1,
        "value": 0.0,
    }


def test_zero_denominators_are_null():
    cases = [_case("stop", "stop", [4], [])]
    trials = plan_trials(cases, repeats=1)
    manifest = _manifest(trials, policies=("first",))
    result = score_result(
        "run",
        "first",
        trials[0],
        cases[0],
        _outcome(Decision(stop=True, dieu=None)),
    )
    full_summary = summarize(manifest, cases, [result])
    summary = full_summary.by_policy["first"]
    assert summary.selection_accuracy.value is None
    assert summary.first_position_selection_rate.value is None
    assert summary.decision_accuracy.value == 1.0
    assert full_summary.paired is None


def test_score_result_rejects_unresolved_labels():
    case = _case("unresolved", "unresolved", [4], [])
    trial = plan_trials([_case("approved", "follow", [4], [4])], repeats=1)[0]
    trial = trial.model_copy(update={"case_id": case.case_id, "observation_hash": case.observation_hash})
    with pytest.raises(ValueError, match="approved follow and stop"):
        score_result("run", "first", trial, case, _outcome(Decision(stop=False, dieu=4)))


def test_summarize_rejects_duplicate_and_outside_schedule_results():
    case = _case("case", "follow", [4, 5], [4])
    trial = plan_trials([case], repeats=1)[0]
    record = score_result("run", "first", trial, case, _outcome(Decision(stop=False, dieu=4)))
    with pytest.raises(ValueError, match="duplicate result key"):
        summarize(_manifest([trial], policies=("first",)), [case], [record, record])
    outsider = trial.model_copy(update={"trial_id": "outside"})
    outside_record = score_result("run", "first", outsider, case, _outcome(Decision(stop=False, dieu=4)))
    with pytest.raises(ValueError, match="unknown trial"):
        summarize(_manifest([trial], policies=("first",)), [case], [outside_record])


def test_mixed_first_position_fixture_uses_independent_position_denominator():
    # Trộn single-candidate, multi-follow và multi-STOP để kiểm tra đúng tập trial được tính
    single = make_case("single", expected_action="follow", candidates=[1], acceptable_dieu={1})
    multi_first = make_case("multi-first", expected_action="follow", candidates=[2, 3], acceptable_dieu={2})
    multi_second = make_case("multi-second", expected_action="follow", candidates=[4, 5], acceptable_dieu={5})
    multi_stop = make_case("multi-stop", expected_action="stop", candidates=[6, 7])
    cases = [single, multi_first, multi_second, multi_stop]
    trials = [make_trial(f"{case.case_id}-r0", case) for case in cases]
    results = [
        score_result("run", "first", trials[0], single, ok_outcome(1)),
        score_result("run", "first", trials[1], multi_first, ok_outcome(2)),
        score_result("run", "first", trials[2], multi_second, ok_outcome(5)),
        score_result("run", "first", trials[3], multi_stop, ok_outcome(None)),
    ]
    summary = summarize(_manifest(trials, policies=("first",)), cases, results).by_policy["first"]

    # Mẫu số chỉ gồm output hợp lệ có ít nhất hai candidates, kể cả STOP
    assert summary.first_position_selection_rate.model_dump() == {
        "numerator": 1,
        "denominator": 3,
        "value": 1 / 3,
    }


def test_error_missing_and_unavailable_usage_keep_accuracy_and_latency_counts():
    # Error và missing vẫn nằm trong mẫu số accuracy nhưng không phải output hợp lệ
    follow = make_case("follow", expected_action="follow", candidates=[1, 2], acceptable_dieu={2})
    stop = make_case("stop", expected_action="stop", candidates=[3, 4])
    missing = make_case("missing", expected_action="follow", candidates=[5, 6], acceptable_dieu={5})
    cases = [follow, stop, missing]
    trials = [make_trial(f"{case.case_id}-r0", case) for case in cases]
    results = [
        score_result("run", "first", trials[0], follow, error_outcome("timeout", latency_ms=7.0)),
        score_result("run", "first", trials[1], stop, ok_outcome(None, latency_ms=3.0)),
    ]
    summary = summarize(_manifest(trials, policies=("first",)), cases, results).by_policy["first"]
    assert summary.selection_accuracy.model_dump() == {"numerator": 0, "denominator": 2, "value": 0.0}
    assert summary.decision_accuracy.model_dump() == {"numerator": 1, "denominator": 3, "value": 1 / 3}
    assert summary.costs["latency_ms"] == {"count": 2, "mean": 5.0, "min": 3.0, "max": 7.0}
    # Usage không được cung cấp phải giữ null, không được diễn giải thành không tốn token
    assert summary.costs["usage"]["total_tokens"] == {"count": 0, "sum": None}


def test_incomplete_pairing_is_excluded_from_paired_correctness():
    # Chỉ so action và article khi cả hai policy có output hợp lệ trên cùng trial
    case = make_case("pair", expected_action="follow", candidates=[1, 2], acceptable_dieu={2})
    trial = make_trial("pair-r0", case)
    first = score_result("run", "first", trial, case, ok_outcome(1))
    llm = score_result("run", "llm", trial, case, error_outcome("transport"))
    paired = summarize(_manifest([trial]), [case], [first, llm]).paired
    assert paired is not None
    assert paired.valid_pairs == 0
    assert paired.incomplete_pairs == 1
    assert paired.agreement.value is None


def test_candidate_groups_and_position_labels_are_reported():
    single = make_case("single", expected_action="follow", candidates=[1], acceptable_dieu={1})
    multi = make_case("multi", expected_action="follow", candidates=[2, 3], acceptable_dieu={3})
    trials = [make_trial("single-r0", single), make_trial("multi-r0", multi)]
    results = [
        score_result("run", "first", trials[0], single, ok_outcome(1)),
        score_result("run", "first", trials[1], multi, ok_outcome(3)),
    ]
    summary = summarize(_manifest(trials, policies=("first",)), [single, multi], results)
    assert summary.by_candidate_group["single_candidate"]["first"].scheduled == 1
    assert summary.by_candidate_group["multi_candidate"]["first"].scheduled == 1
    assert summary.by_position["1"]["first"].value == 0.0
    assert summary.by_position["2"]["first"].value == 1.0


def test_plan_trials_preserves_repeat_identity_and_schedule():
    case = make_case("repeat", expected_action="follow", candidates=[1, 2], acceptable_dieu={1})
    trials = plan_trials([case], repeats=2)
    assert [trial.repeat_id for trial in trials] == [0, 1]
    assert len({trial.trial_id for trial in trials}) == 2
    assert all(trial.candidate_order == [1, 2] for trial in trials)
    assert all(trial.permutation_id == "original" for trial in trials)


def test_score_result_uses_the_recorded_candidate_order_for_position_and_correctness():
    case = make_case(
        "rotated",
        expected_action="follow",
        candidates=[2, 3],
        acceptable_dieu={3},
    )
    trial = Trial(
        trial_id="rotated:candidate-order:c[3,2]:r0",
        case_id=case.case_id,
        repeat_id=0,
        condition="candidate-order",
        permutation_id="c[3,2]",
        seed_order=[],
        candidate_order=[3, 2],
        observation_hash=case.observation_hash,
    )
    result = score_result(
        "run",
        "first",
        trial,
        case,
        ok_outcome(3),
    )
    assert result.selected_position == 1
    assert result.selection_correct is True
    assert result.decision_correct is True
