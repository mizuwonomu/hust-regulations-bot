"""Kiểm tra lịch replay STOP tất định, input trace và fingerprint của nó"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stop_policy_synthetic import retarget, reviewed_corpus

from stop_policy_eval.cases import label_free_input, write_new_stop_cases
from stop_policy_eval.artifacts import load_run, prepare_run
from stop_policy_eval.contracts import BASELINE_VARIANT
from stop_policy_eval.execution import render_baseline_trace
from stop_policy_eval.schedule import (
    VALID_POLICIES,
    VARIANT_RECIPES,
    build_hop0_capture_plan,
    build_stop_plan,
    plan_selected_trials,
    resolve_variant,
)


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory):
    """Review bundle tổng hợp đủ bốn lớp nhãn trên hai split"""
    return reviewed_corpus(tmp_path_factory.mktemp("schedule"), name="schedule")


def _plan(reviewed, *, split="dev", policies=("first", "llm"), repeats=3, variant=BASELINE_VARIANT):
    return build_stop_plan(
        list(reviewed.cases),
        reviewed.snapshot,
        reviewed.corpus,
        cases_path=reviewed.reviewed_path,
        snapshot_path=reviewed.seeds_path,
        split=split,
        policies=list(policies),
        repeats=repeats,
        variant=variant,
    )


def test_only_approved_overlap_clear_cases_of_the_split_are_scheduled(reviewed):
    plan = _plan(reviewed)
    scheduled = {trial.case_id for trial in plan.trials}
    by_id = {case.case_id: case for case in reviewed.cases}
    assert scheduled
    for case_id in scheduled:
        case = by_id[case_id]
        assert case.label_status == "approved"
        assert case.fewshot_overlap is False
        assert case.split == "dev"
        assert case.expected_action in {"follow", "stop"}
        assert case.candidates
    labels = {by_id[case_id].expected_action for case_id in scheduled}
    assert labels == {"follow", "stop"}
    assert plan.readiness.count_for("hop0_follow") == 1
    assert plan.readiness.count_for("post_follow_follow") == 1
    assert plan.readiness.complete is True


def test_the_other_split_is_excluded_with_a_distinct_reason(reviewed):
    plan = _plan(reviewed, split="dev")
    by_id = {case.case_id: case for case in reviewed.cases}
    other_split = {
        case.case_id for case in reviewed.cases if case.split == "heldout" and case.label_status == "approved"
    }
    reasons = {item.case_id: item.reason for item in plan.exclusions}
    assert other_split
    for case_id in other_split:
        assert reasons[case_id] == "split_not_selected"
    assert {by_id[item.case_id].split for item in plan.exclusions if item.reason == "draft"} == {"unassigned"}
    assert plan.selected_case_ids == [
        case.case_id for case in reviewed.approved if case.split == "dev"
    ]


def test_hop0_capture_plan_schedules_nonempty_drafts_without_full_readiness(reviewed, tmp_path):
    source_cases = [
        case.model_copy(
            update={
                "label_status": "draft",
                "label_reason": None,
                "label_observation_hash": None,
                "fewshot_overlap": None,
                "split": "dev",
            }
        )
        for case in reviewed.cases
        if case.source_hop == 0
    ]
    cases_path = tmp_path / "hop0-source-cases.jsonl"
    write_new_stop_cases(cases_path, source_cases)

    plan = build_hop0_capture_plan(
        source_cases,
        reviewed.snapshot,
        reviewed.corpus,
        cases_path=cases_path,
        snapshot_path=reviewed.seeds_path,
        split="dev",
    )

    assert plan.run_purpose == "later_hop_capture"
    assert plan.policies == ["llm"]
    assert plan.repeats == 1
    assert plan.expected_trial_count == sum(bool(case.candidates) for case in source_cases)
    assert all(trace.source_hop == 0 and trace.candidates for trace in plan.traces)
    assert plan.readiness.complete is False
    run_dir = tmp_path / "hop0-source-run"
    manifest = prepare_run(plan, run_dir)
    loaded_manifest, _, traces, results = load_run(run_dir)
    assert loaded_manifest == manifest
    assert loaded_manifest.run_purpose == "later_hop_capture"
    assert len(traces) == manifest.expected_trial_count
    assert results == []


def test_repeats_create_stable_unique_trial_ids_from_case_identity(reviewed):
    with pytest.raises(ValueError):
        _plan(reviewed, repeats=0)
    with pytest.raises(ValueError):
        _plan(reviewed, repeats=True)
    plan = _plan(reviewed, repeats=3)
    for case_id in plan.selected_case_ids:
        trials = [trial for trial in plan.trials if trial.case_id == case_id]
        assert [trial.repeat_id for trial in trials] == [0, 1, 2]
        assert [trial.trial_id for trial in trials] == [
            f"{case_id}:r0",
            f"{case_id}:r1",
            f"{case_id}:r2",
        ]
    assert plan.expected_trial_count == len(plan.selected_case_ids) * 3
    assert plan.expected_policy_result_count == plan.expected_trial_count * 2
    assert plan == _plan(reviewed, repeats=3)


def test_case_order_follows_the_case_artifact_not_the_label(reviewed):
    plan = _plan(reviewed)
    expected = [case.case_id for case in reviewed.approved if case.split == "dev"]
    seen: list[str] = []
    for trial in plan.trials:
        if not seen or seen[-1] != trial.case_id:
            seen.append(trial.case_id)
    assert seen == expected

    reversed_plan = build_stop_plan(
        list(reversed(list(reviewed.cases))),
        reviewed.snapshot,
        reviewed.corpus,
        cases_path=reviewed.reviewed_path,
        snapshot_path=reviewed.seeds_path,
        split="dev",
        policies=["first"],
        repeats=1,
        variant=BASELINE_VARIANT,
    )
    assert [trial.case_id for trial in reversed_plan.trials] == list(reversed(expected))


def test_one_trace_per_case_is_repeat_independent(reviewed):
    plan = _plan(reviewed, repeats=4)
    assert len(plan.traces) == len(plan.selected_case_ids)
    assert {trace.input_trace_id for trace in plan.traces} == {
        f"{case_id}:{BASELINE_VARIANT}" for case_id in plan.selected_case_ids
    }
    for trace in plan.traces:
        trials = [trial for trial in plan.trials if trial.case_id == trace.case_id]
        assert len(trials) == 4
        assert {trial.input_fingerprint for trial in trials} == {trace.input_fingerprint}


def test_baseline_messages_and_grammar_match_the_production_renderers(reviewed):
    from src.rag.agent.prompt import build_decision_grammar, render_gate_prompt

    plan = _plan(reviewed, repeats=1)
    case_by_id = {case.case_id: case for case in reviewed.cases}
    for trace in plan.traces:
        case = case_by_id[trace.case_id]
        rendered = render_gate_prompt(case.question, case.observation, list(case.candidates))
        expected = [(message.type, message.content) for message in rendered.to_messages()]
        assert [(message.role, message.content) for message in trace.effective_messages] == expected
        assert trace.grammar_text == build_decision_grammar(list(case.candidates))
        assert trace.question == case.question
        assert trace.observation == case.observation
        assert trace.candidates == list(case.candidates)
        assert trace.grammar_candidates == list(case.candidates)


def test_traces_carry_no_labels_or_corpus_annotations(reviewed):
    plan = _plan(reviewed, repeats=1)
    case_by_id = {case.case_id: case for case in reviewed.cases}
    for trace in plan.traces:
        payload = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)
        case = case_by_id[trace.case_id]
        assert case.label_reason not in payload
        assert "acceptable_dieu" not in payload
        assert "expected_action" not in payload
        assert "CORPUS_ANSWER_ONLY_MARKER" not in payload
        assert "CORPUS_GOLD_CONTEXT_MARKER" not in payload
        assert case.semantic_group_id not in payload


def test_input_fingerprint_binds_exact_inputs_and_variant(reviewed):
    plan = _plan(reviewed, repeats=1)
    trace = next(item for item in plan.traces if item.case_id == reviewed.case_for(1).case_id)
    case = reviewed.case_for(1)
    assert trace.input_fingerprint == render_baseline_trace(case, BASELINE_VARIANT).input_fingerprint

    renewed = retarget(reviewed.case_for(1, source_hop=1), observation="Observation khác hoàn toàn\n")
    assert (
        render_baseline_trace(renewed, BASELINE_VARIANT).input_fingerprint
        != render_baseline_trace(reviewed.case_for(1, source_hop=1), BASELINE_VARIANT).input_fingerprint
    )


def test_unsupported_variants_are_rejected_before_materialization(reviewed):
    assert resolve_variant(BASELINE_VARIANT) == BASELINE_VARIANT
    assert VARIANT_RECIPES[BASELINE_VARIANT]
    for variant in ("prompt-balanced", "zero-shot", "observation-context", "thinking-on", ""):
        with pytest.raises(ValueError, match="variant"):
            resolve_variant(variant)
        with pytest.raises(ValueError, match="variant"):
            _plan(reviewed, variant=variant)


def test_plan_and_trace_reject_unsupported_policies_and_bad_counts(reviewed):
    assert set(VALID_POLICIES) == {"first", "llm"}
    with pytest.raises(ValueError, match="polic"):
        _plan(reviewed, policies=("first", "first"))
    with pytest.raises(ValueError, match="polic"):
        _plan(reviewed, policies=("first", "judge"))
    with pytest.raises(ValueError, match="polic"):
        _plan(reviewed, policies=())


def test_preparation_refuses_a_split_missing_a_required_class(reviewed):
    hop0_only = [case for case in reviewed.cases if case.source_hop == 0]
    with pytest.raises(ValueError, match="post_follow"):
        build_stop_plan(
            hop0_only,
            reviewed.snapshot,
            reviewed.corpus,
            cases_path=reviewed.reviewed_path,
            snapshot_path=reviewed.seeds_path,
            split="dev",
            policies=["first"],
            repeats=1,
            variant=BASELINE_VARIANT,
        )


def test_plan_sources_are_hashed_and_repository_relative_inside_the_repo(reviewed, tmp_path, monkeypatch):
    import stop_policy_eval.cases as cases_module

    plan = _plan(reviewed, repeats=1)
    assert plan.snapshot_source.sha256 == hashlib.sha256(reviewed.seeds_path.read_bytes()).hexdigest()
    assert plan.cases_source.sha256 == hashlib.sha256(reviewed.reviewed_path.read_bytes()).hexdigest()
    assert plan.corpus_sha256 == reviewed.corpus.sha256
    # Nguồn ngoài repository vẫn resolve được về đúng bytes đã băm
    assert Path(cases_module.resolve_source_file(plan.cases_source)) == reviewed.reviewed_path

    monkeypatch.setattr(cases_module, "repository_root", lambda: reviewed.reviewed_path.parent)
    inside = _plan(reviewed, repeats=1)
    assert not inside.cases_source.path.startswith("/")
    assert inside.cases_source.path == reviewed.reviewed_path.name
    assert Path(cases_module.resolve_source_file(inside.cases_source)) == reviewed.reviewed_path


def test_trials_are_derivable_from_the_approved_cases(reviewed):
    plan = _plan(reviewed, repeats=2)
    rebuilt = plan_selected_trials(
        list(reviewed.cases),
        reviewed.snapshot,
        split="dev",
        repeats=2,
        variant=BASELINE_VARIANT,
    )
    assert [trial.model_dump(mode="json") for trial in rebuilt] == [
        trial.model_dump(mode="json") for trial in plan.trials
    ]


def test_traces_do_not_store_labels_even_for_a_hop_case(reviewed):
    trace = render_baseline_trace(reviewed.case_for(1, source_hop=1), BASELINE_VARIANT)
    assert trace.hop_class == "post_follow"
    assert trace.source_hop == 1
    assert trace.question == reviewed.case_for(1, source_hop=1).question


def test_gate_input_projection_is_label_free(reviewed):
    case = reviewed.case_for(1)
    projection = label_free_input(case)
    assert projection.question == case.question
    assert projection.observation == case.observation
    assert projection.candidates == list(case.candidates)
    payload = json.dumps(projection.model_dump(mode="json"), ensure_ascii=False)
    for forbidden in ("acceptable", "expected_action", "label", "split", "semantic_group"):
        assert forbidden not in payload
