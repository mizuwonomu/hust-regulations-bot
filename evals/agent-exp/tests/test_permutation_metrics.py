"""Oracle độc lập cho summary của permutation replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from contracts import PermutationConfig, RunManifest, SourceFile
from metrics import score_result
from permutation import build_permutation_plan
from permutation_metrics import summarize_permutation
from synthetic import error_outcome, make_case, ok_outcome


def _manifest(trials, *, config, policies=("first",)):
    return RunManifest(
        schema_version=2,
        run_id="run",
        experiment="permutation",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at=None,
        status="running",
        policies=list(policies),
        trials=trials,
        cases_source=SourceFile(path="cases.jsonl", sha256="cases"),
        snapshot_source=SourceFile(path="snapshot.json", sha256="snapshot"),
        exclusions=[],
        provenance={},
        policy_config={policy: {} for policy in policies},
        permutation=config,
        expected_trial_count=len(trials),
        expected_policy_result_count=len(trials) * len(policies),
    )


def _candidate_plan(case, *, repeats=1):
    config = PermutationConfig(
        condition="candidate-order", schedule="rotate", repeats=repeats
    )
    plan = build_permutation_plan([case], _snapshot(), config)
    return config, plan


def _repeat_plan(case, *, repeats=3):
    config = PermutationConfig(condition="repeat", schedule="original", repeats=repeats)
    plan = build_permutation_plan([case], _snapshot(), config)
    return config, plan


def _snapshot():
    from contracts import SeedRow, SeedSnapshot

    return SeedSnapshot(
        schema_version=1,
        snapshot_id="snapshot",
        dataset_id="synthetic",
        baseline_source=SourceFile(path="baseline", sha256="baseline"),
        dataset_source=SourceFile(path="dataset", sha256="dataset"),
        whitelist_source=SourceFile(path="whitelist", sha256="whitelist"),
        retrieval_config={},
        unavailable_metadata={},
        internal_dieu={1, 2, 3, 4, 5, 6, 7},
        rows=[
            SeedRow(
                id=1,
                question="question",
                contexts=["seed"],
                seed_dieu={1},
            )
        ],
    )


def _records(manifest, cases, outcomes_by_trial, *, policy="first"):
    case_by_id = {case.case_id: case for case in cases}
    return [
        score_result(
            manifest.run_id,
            policy,
            trial,
            case_by_id[trial.case_id],
            outcomes_by_trial[trial.trial_id],
        )
        for trial in manifest.trials
        if trial.trial_id in outcomes_by_trial
    ]


def test_correct_id_across_rotations_has_accuracy_and_consistency_one():
    case = make_case("correct", expected_action="follow", candidates=[1, 2, 3], acceptable_dieu={1})
    config, plan = _candidate_plan(case)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {trial.trial_id: ok_outcome(1) for trial in plan.trials}
    summary = summarize_permutation(manifest, [case], _records(manifest, [case], outcomes))
    condition = summary.by_policy["first"].by_condition["candidate-order"]
    assert condition.selection_accuracy.model_dump() == {"numerator": 3, "denominator": 3, "value": 1.0}
    assert condition.permutation_consistency.pooled.model_dump() == {
        "numerator": 3,
        "denominator": 3,
        "value": 1.0,
    }


def test_first_candidate_has_one_over_n_accuracy_and_zero_permutation_consistency():
    case = make_case("first", expected_action="follow", candidates=[1, 2, 3], acceptable_dieu={1})
    config, plan = _candidate_plan(case)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {
        trial.trial_id: ok_outcome(trial.candidate_order[0]) for trial in plan.trials
    }
    summary = summarize_permutation(manifest, [case], _records(manifest, [case], outcomes))
    condition = summary.by_policy["first"].by_condition["candidate-order"]
    assert condition.selection_accuracy.value == pytest.approx(1 / 3)
    assert condition.permutation_consistency.pooled.value == 0.0
    assert condition.first_position_selection_rate.value == 1.0


def test_always_stop_is_consistent_but_wrong_and_keeps_stop_in_position_denominator():
    case = make_case("stop", expected_action="follow", candidates=[1, 2, 3], acceptable_dieu={1})
    config, plan = _candidate_plan(case)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {trial.trial_id: ok_outcome(None) for trial in plan.trials}
    summary = summarize_permutation(manifest, [case], _records(manifest, [case], outcomes))
    condition = summary.by_policy["first"].by_condition["candidate-order"]
    assert condition.selection_accuracy.value == 0.0
    assert condition.permutation_consistency.pooled.value == 1.0
    assert condition.first_position_selection_rate.model_dump() == {
        "numerator": 0,
        "denominator": 3,
        "value": 0.0,
    }
    assert condition.position_distributions[0].stop.value == 1.0


def test_stable_wrong_id_has_consistency_one_and_accuracy_zero():
    case = make_case("wrong", expected_action="follow", candidates=[1, 2, 3], acceptable_dieu={1})
    config, plan = _candidate_plan(case)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {trial.trial_id: ok_outcome(2) for trial in plan.trials}
    summary = summarize_permutation(manifest, [case], _records(manifest, [case], outcomes))
    condition = summary.by_policy["first"].by_condition["candidate-order"]
    assert condition.selection_accuracy.value == 0.0
    assert condition.permutation_consistency.pooled.value == 1.0


def test_repeat_consistency_is_separate_from_permutation_consistency():
    # Repeat consistency dùng cùng input, còn permutation consistency không có rotation
    case = make_case("repeat", expected_action="follow", candidates=[1, 2], acceptable_dieu={1})
    config, plan = _repeat_plan(case, repeats=3)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {trial.trial_id: ok_outcome(1) for trial in plan.trials}
    summary = summarize_permutation(manifest, [case], _records(manifest, [case], outcomes))
    condition = summary.by_policy["first"].by_condition["repeat"]
    assert condition.repeat_consistency.pooled.value == 1.0
    assert condition.repeat_consistency.case_mean.value == 1.0
    assert condition.cases[0].repeat_consistency.value == 1.0
    assert condition.question_macro.repeat_consistency.value == 1.0
    assert condition.permutation_consistency.pooled.value is None


def test_missing_and_error_outputs_use_scheduled_pair_denominators():
    case = make_case("partial", expected_action="follow", candidates=[1, 2, 3], acceptable_dieu={1})
    config, plan = _candidate_plan(case)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {
        plan.trials[0].trial_id: ok_outcome(1),
        plan.trials[1].trial_id: error_outcome("timeout"),
    }
    summary = summarize_permutation(manifest, [case], _records(manifest, [case], outcomes))
    condition = summary.by_policy["first"].by_condition["candidate-order"]
    assert condition.scheduled == 3
    assert condition.valid == 1
    assert condition.errors == 1
    assert condition.missing == 1
    assert condition.selection_accuracy.model_dump() == {
        "numerator": 1,
        "denominator": 3,
        "value": 1 / 3,
    }
    assert condition.permutation_consistency.scheduled_pairs == 3
    assert condition.permutation_consistency.valid_pairs == 0
    assert condition.permutation_consistency.pooled.value is None


def test_case_and_question_macros_do_not_become_pooled_trial_ratios():
    first = make_case("q1-hop0", expected_action="follow", candidates=[1, 2], acceptable_dieu={1})
    first = first.model_copy(update={"question_id": 1})
    second = make_case("q1-hop1", expected_action="follow", candidates=[3, 4], acceptable_dieu={3})
    second = second.model_copy(update={"question_id": 1, "source_hop": 1})
    third = make_case("q2", expected_action="follow", candidates=[5, 6, 7], acceptable_dieu={5})
    third = third.model_copy(update={"question_id": 2})
    cases = [first, second, third]
    config = PermutationConfig(condition="candidate-order", schedule="rotate", repeats=1)
    plan = build_permutation_plan(cases, _snapshot(), config)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {trial.trial_id: ok_outcome(trial.candidate_order[0]) for trial in plan.trials}
    summary = summarize_permutation(manifest, cases, _records(manifest, cases, outcomes))
    condition = summary.by_policy["first"].by_condition["candidate-order"]
    assert condition.selection_accuracy.model_dump() == {
        "numerator": 3,
        "denominator": 7,
        "value": 3 / 7,
    }
    assert condition.selection_case_macro.value == pytest.approx((0.5 + 0.5 + 1 / 3) / 3)
    assert condition.question_macro.selection_accuracy.value == pytest.approx((0.5 + 1 / 3) / 2)
    assert condition.question_macro.question_count == 2


def test_integer_and_string_question_ids_remain_distinct():
    integer_case = make_case("int", expected_action="follow", candidates=[1, 2], acceptable_dieu={1})
    string_case = make_case("string", expected_action="follow", candidates=[1, 2], acceptable_dieu={1})
    integer_case = integer_case.model_copy(update={"question_id": 1})
    string_case = string_case.model_copy(update={"question_id": "1"})
    config = PermutationConfig(condition="candidate-order", schedule="rotate", repeats=1)
    plan = build_permutation_plan([integer_case, string_case], _snapshot(), config)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {trial.trial_id: ok_outcome(1) for trial in plan.trials}
    summary = summarize_permutation(
        manifest,
        [integer_case, string_case],
        _records(manifest, [integer_case, string_case], outcomes),
    )
    condition = summary.by_policy["first"].by_condition["candidate-order"]
    assert condition.question_macro.question_count == 2


def test_summary_is_invariant_to_result_row_order_and_recomputes_correctness():
    # Summary phải bỏ qua thứ tự JSONL và tự tính lại correctness từ decision
    case = make_case("order", expected_action="follow", candidates=[1, 2], acceptable_dieu={2})
    config, plan = _candidate_plan(case)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {trial.trial_id: ok_outcome(1) for trial in plan.trials}
    records = _records(manifest, [case], outcomes)
    corrupted = [record.model_copy(update={"selection_correct": True, "decision_correct": True}) for record in records]
    original = summarize_permutation(manifest, [case], records).model_dump(mode="json")
    recomputed = summarize_permutation(manifest, [case], list(reversed(corrupted))).model_dump(mode="json")
    assert recomputed == original
    assert recomputed["by_policy"]["first"]["by_condition"]["candidate-order"]["selection_accuracy"]["value"] == 0.0


def test_identical_inputs_include_ordered_candidates_and_group_equal_full_inputs():
    # Combined phải tách candidate order nhưng gom seed order có cùng input đầy đủ
    from artifacts import sha256_text
    from contracts import SeedRow, SeedSnapshot
    from seed_cases import rebuild_case_state

    row_text = "Điều 1. Nguồn\nTheo Điều 2 và Điều 3"
    snapshot = SeedSnapshot(
        schema_version=1,
        snapshot_id="snapshot",
        dataset_id="synthetic",
        baseline_source=SourceFile(path="baseline", sha256="baseline"),
        dataset_source=SourceFile(path="dataset", sha256="dataset"),
        whitelist_source=SourceFile(path="whitelist", sha256="whitelist"),
        retrieval_config={},
        unavailable_metadata={},
        internal_dieu={1, 2, 3},
        rows=[
            SeedRow(
                id=1,
                question="question",
                contexts=[row_text, row_text],
                seed_dieu={1},
            )
        ],
    )
    observation, candidates = rebuild_case_state(snapshot, snapshot.rows[0])
    case = make_case(
        "combined-inputs",
        expected_action="follow",
        candidates=list(candidates),
        acceptable_dieu={candidates[0]},
    ).model_copy(
        update={
            "question_id": 1,
            "snapshot_id": snapshot.snapshot_id,
            "observation": observation,
            "observation_hash": sha256_text(observation),
        }
    )
    config = PermutationConfig(condition="combined", schedule="rotate", repeats=1)
    plan = build_permutation_plan([case], snapshot, config)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {
        trial.trial_id: ok_outcome(trial.candidate_order[0]) for trial in plan.trials
    }
    summary = summarize_permutation(manifest, [case], _records(manifest, [case], outcomes))
    groups = summary.by_policy["first"].by_condition["combined"].identical_inputs

    assert len(plan.trials) == 4
    assert len(groups) == 2
    assert {tuple(group.candidate_order) for group in groups} == {
        tuple(candidates),
        tuple(reversed(candidates)),
    }
    assert all(len(group.permutation_ids) == 2 for group in groups)


def test_consistency_macro_averages_case_values_then_question_values():
    # Fixture độc lập tạo case macro 1/2 và question macro 2/3
    cases = []
    for index, value in enumerate((1, 0, 0), start=1):
        cases.append(
            make_case(
                f"question-a-{index}",
                expected_action="follow",
                candidates=[1, 2],
                acceptable_dieu={1},
            ).model_copy(update={"question_id": "question-a"})
        )
    cases.append(
        make_case(
            "question-b-1",
            expected_action="follow",
            candidates=[1, 2],
            acceptable_dieu={1},
        ).model_copy(update={"question_id": "question-b"})
    )
    config = PermutationConfig(condition="candidate-order", schedule="rotate", repeats=1)
    plan = build_permutation_plan(cases, _snapshot(), config)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {}
    for trial in plan.trials:
        case_index = next(
            index for index, case in enumerate(cases) if case.case_id == trial.case_id
        )
        stable = case_index in {0, 3}
        outcomes[trial.trial_id] = ok_outcome(
            1 if stable else trial.candidate_order[0]
        )
    summary = summarize_permutation(manifest, cases, _records(manifest, cases, outcomes))
    condition = summary.by_policy["first"].by_condition["candidate-order"]
    case_values = {
        item.case_id: item.permutation_consistency.value for item in condition.cases
    }

    assert list(case_values.values()) == [1.0, 0.0, 0.0, 1.0]
    assert condition.permutation_consistency.case_mean.value == pytest.approx(0.5)
    assert condition.question_macro.permutation_consistency.value == pytest.approx(2 / 3)
    assert condition.question_macro.permutation_consistency.defined == 2
    assert condition.question_macro.permutation_consistency.eligible == 2


def test_gold_position_accuracy_uses_each_trial_candidate_order():
    # Gold ở vị trí đầu và cuối phải vào hai bucket accuracy khác nhau
    case = make_case("gold-position", expected_action="follow", candidates=[1, 2], acceptable_dieu={1})
    config, plan = _candidate_plan(case)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {
        trial.trial_id: ok_outcome(trial.candidate_order[0]) for trial in plan.trials
    }
    summary = summarize_permutation(manifest, [case], _records(manifest, [case], outcomes))
    groups = summary.by_policy["first"].by_condition["candidate-order"].gold_positions
    by_position = {group.gold_positions[0]: group for group in groups}

    assert by_position[1].accuracy.model_dump() == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }
    assert by_position[2].accuracy.model_dump() == {
        "numerator": 0,
        "denominator": 1,
        "value": 0.0,
    }


def test_position_distribution_excludes_errors_from_valid_output_rates():
    # Error nằm trong accuracy scheduled nhưng không được tính là STOP
    case = make_case("valid-position", expected_action="follow", candidates=[1, 2], acceptable_dieu={1})
    config, plan = _candidate_plan(case)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {
        plan.trials[0].trial_id: ok_outcome(plan.trials[0].candidate_order[0]),
        plan.trials[1].trial_id: error_outcome("transport"),
    }
    summary = summarize_permutation(manifest, [case], _records(manifest, [case], outcomes))
    condition = summary.by_policy["first"].by_condition["candidate-order"]
    distribution = next(
        item for item in condition.position_distributions if item.candidate_count == 2
    )

    assert condition.selection_accuracy.denominator == 2
    assert distribution.coverage.model_dump() == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert distribution.positions["1"].model_dump() == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }
    assert distribution.stop.model_dump() == {
        "numerator": 0,
        "denominator": 1,
        "value": 0.0,
    }


def test_consistency_group_identity_retains_case_boundaries():
    # Hai case dùng cùng repeat và permutation id vẫn giữ đủ pair của từng case
    stable = make_case("stable-case", expected_action="follow", candidates=[1, 2], acceptable_dieu={1})
    unstable = make_case("unstable-case", expected_action="follow", candidates=[1, 2], acceptable_dieu={1})
    config = PermutationConfig(condition="candidate-order", schedule="rotate", repeats=2)
    cases = [stable, unstable]
    plan = build_permutation_plan(cases, _snapshot(), config)
    manifest = _manifest(plan.trials, config=config)
    outcomes = {}
    for trial in plan.trials:
        if trial.case_id == stable.case_id:
            outcomes[trial.trial_id] = ok_outcome(1)
        else:
            outcomes[trial.trial_id] = ok_outcome(trial.candidate_order[0])
    summary = summarize_permutation(manifest, cases, _records(manifest, cases, outcomes))
    consistency = summary.by_policy["first"].by_condition["candidate-order"].permutation_consistency

    assert consistency.matching_pairs == 2
    assert consistency.valid_pairs == 4
    assert consistency.scheduled_pairs == 4
