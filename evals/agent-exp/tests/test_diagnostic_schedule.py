"""Kiểm tra offline lịch trình diagnostic A/B và hash nguồn tươi của fresh run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT_EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_EXP_ROOT / "scripts"))
sys.path.insert(0, str(AGENT_EXP_ROOT.parents[1]))

from artifacts import read_snapshot
from seed_cases import read_cases

from diagnostic_subexp.article_42_observation_block_order.contracts import ObservationSuiteSpec
from diagnostic_subexp.article_42_observation_block_order.schedule import (
    build_observation_definition,
    plan_observation_run,
    read_observation_spec,
    resolve_observation_sources,
)
from diagnostic_subexp.candidate_pair_41_3_position.contracts import CandidateSuiteSpec
from diagnostic_subexp.candidate_pair_41_3_position.schedule import (
    build_candidate_definition,
    plan_candidate_run,
    question_identity,
    read_candidate_spec,
    resolve_candidate_sources,
    validate_candidate_arm_schedule,
)
from diagnostic_subexp.shared.source_refs import sha256_file


def _payload(definition: dict) -> dict:
    return json.loads(json.dumps(definition))


def _load_a(policies: tuple[str, ...] = ("first", "llm")):
    spec = read_candidate_spec(None)
    sources = resolve_candidate_sources(spec)
    cases = read_cases(sources["case"])
    snapshot = read_snapshot(sources["snapshot"])
    plan = plan_candidate_run(spec, cases, snapshot, list(policies))
    return spec, cases, snapshot, plan


def _load_b(policies: tuple[str, ...] = ("first", "llm")):
    spec = read_observation_spec(None)
    sources = resolve_observation_sources(spec)
    cases = read_cases(sources["case"])
    snapshot = read_snapshot(sources["snapshot"])
    plan = plan_observation_run(spec, cases, snapshot, list(policies))
    return spec, cases, snapshot, plan


def test_definition_is_built_in_memory_without_source_hashes():
    """Definition mới không được chứa hash nguồn lịch sử hay identity treatment C."""
    for definition in (build_candidate_definition(), build_observation_definition()):
        serialized = json.dumps(definition)
        for forbidden in (
            "case_hash",
            "snapshot_hash",
            "inventory_hash",
            "prompt_source_hash",
            "prompt_variant_id",
            "source_revision",
            "snapshot_id",
            "migration",
        ):
            assert forbidden not in serialized


def test_plan_records_fresh_source_hashes_read_from_canonical_bytes():
    """Plan ghi hash đúng bằng sha256 của file canonical đọc tại prepare."""
    spec, _, _, plan = _load_a()
    assert plan.case_source.sha256 == sha256_file(AGENT_EXP_ROOT.parents[1] / spec.case_path)
    assert plan.snapshot_source.sha256 == sha256_file(
        AGENT_EXP_ROOT.parents[1] / spec.snapshot_path
    )
    assert plan.inventory_source.sha256 == sha256_file(
        AGENT_EXP_ROOT.parents[1] / spec.inventory_path
    )
    assert all(reference.required_for_replay for reference in (
        plan.case_source,
        plan.snapshot_source,
        plan.inventory_source,
    ))


def test_a_spec_materializes_the_declared_schedule():
    spec, _, _, plan = _load_a()
    assert plan.diagnostic_kind == "candidate-relative-position"
    assert len(plan.traces) == 5
    assert plan.expected_trial_count == 30
    assert plan.expected_policy_result_count == 60
    by_arm = {arm.arm_id: arm for arm in spec.arms}
    assert {trial.arm_id for trial in plan.trials} == set(by_arm)
    for trial in plan.trials:
        assert trial.presented_candidates == by_arm[trial.arm_id].presented_candidates
        assert trial.grammar_candidates == spec.grammar_candidates
    batch_zero = [trial for trial in plan.trials if trial.batch_id == "batch-0"]
    assert [trial.arm_id for trial in batch_zero[:5]] == spec.batches[0].arm_order
    batch_one = [trial for trial in plan.trials if trial.batch_id == "batch-1"]
    assert [trial.arm_id for trial in batch_one[:5]] == spec.batches[1].arm_order


def test_a_schedule_balances_pair_orientation_per_slot_stratum():
    spec, _, _, _ = _load_a()
    strata = validate_candidate_arm_schedule(spec.arms)
    assert [stratum["pair_positions"] for stratum in strata] == [[2, 3], [3, 4]]
    for stratum in strata:
        assert len(stratum["arms"]) == 2
        assert stratum["orientations"] == ["3-41", "41-3"]
    for arm in spec.arms:
        if arm.matched_pair_id is None:
            continue
        positions = arm.pair_positions
        assert positions[1] - positions[0] == 1


def test_a_schedule_excludes_control_from_matched_counts():
    spec, _, _, _ = _load_a()
    strata = validate_candidate_arm_schedule(spec.arms)
    matched = {arm_id for stratum in strata for arm_id in stratum["arms"]}
    assert "a0-original-control" not in matched
    assert matched == {
        "a1-pair-23-3-first",
        "a2-pair-23-41-first",
        "a3-pair-34-3-first",
        "a4-pair-34-41-first",
    }


def test_batch_and_repeat_expansion_only_repeats_declared_configurations():
    spec, _, _, plan = _load_a()
    scheduled = {tuple(trial.presented_candidates) for trial in plan.trials}
    declared = {tuple(arm.presented_candidates) for arm in spec.arms}
    assert scheduled == declared
    assert len({(trial.arm_id, tuple(trial.presented_candidates)) for trial in plan.trials}) == len(
        spec.arms
    )
    for arm in spec.arms:
        arm_trials = [trial for trial in plan.trials if trial.arm_id == arm.arm_id]
        assert len(arm_trials) == 6


def test_a_schedule_rejects_duplicate_orientation():
    spec, _, _, _ = _load_a()
    duplicate = spec.arms[2].model_copy(update={"relative_pair_order": "3-41"})
    broken = spec.model_copy(update={"arms": [*spec.arms[:2], duplicate, *spec.arms[3:]]})
    with pytest.raises(ValueError, match="one orientation of each pair order"):
        validate_candidate_arm_schedule(broken.arms)


def test_a_schedule_rejects_moved_distractor_inside_a_matched_swap():
    spec, _, _, _ = _load_a()
    moved = spec.arms[2].model_copy(update={"presented_candidates": [8, 41, 3, 43, 40]})
    broken = spec.model_copy(update={"arms": [*spec.arms[:2], moved, *spec.arms[3:]]})
    with pytest.raises(ValueError, match="non-pair candidate"):
        validate_candidate_arm_schedule(broken.arms)


def test_a_spec_rejects_changed_candidate_membership():
    payload = _payload(build_candidate_definition())
    payload["arms"][1]["presented_candidates"] = [8, 3, 41, 40, 44]
    with pytest.raises(ValueError, match="changes candidate membership"):
        CandidateSuiteSpec.model_validate(payload)


def test_a_spec_rejects_a_non_reversed_second_batch():
    payload = _payload(build_candidate_definition())
    payload["batches"][1]["arm_order"] = list(payload["batches"][0]["arm_order"])
    with pytest.raises(ValueError, match="batch 1 must reverse"):
        CandidateSuiteSpec.model_validate(payload)


def test_b_spec_materializes_two_traces_and_twelve_trials():
    spec, _, _, plan = _load_b()
    assert plan.diagnostic_kind == "observation-block-order"
    assert len(plan.traces) == 2
    assert plan.expected_trial_count == 12
    assert plan.expected_policy_result_count == 24
    for trace in plan.traces:
        assert trace.presented_candidates == spec.grammar_candidates
        assert trace.original_observation is not None
        assert trace.transformed_observation is not None
    assert {trial.arm_id for trial in plan.trials} == {
        "b0-original-control",
        "b1-reversed-blocks",
    }


def test_b_arms_must_keep_the_original_candidate_order():
    payload = _payload(build_observation_definition())
    payload["arms"][1]["presented_candidates"] = [8, 41, 3, 40, 43]
    with pytest.raises(ValueError, match="keep the original candidate order"):
        ObservationSuiteSpec.model_validate(payload)


def test_question_identity_is_typed_not_coerced():
    assert question_identity(5) != question_identity("5")
    spec, cases, snapshot, _ = _load_a()
    identity = spec.source_case_identity.model_copy(update={"question_id": "5"})
    broken = spec.model_copy(update={"source_case_identity": identity})
    with pytest.raises(ValueError, match="question id does not match"):
        plan_candidate_run(broken, cases, snapshot, ["first"])


def test_plan_rejects_a_case_snapshot_hash_drift():
    """Case ghi snapshot hash khác canonical snapshot phải bị từ chối."""
    spec, cases, snapshot, _ = _load_a()
    target = next(case for case in cases if case.case_id == spec.source_case_identity.case_id)
    drifted = target.model_copy(update={"snapshot_hash": "0" * 64})
    broken_cases = [drifted if case.case_id == target.case_id else case for case in cases]
    with pytest.raises(ValueError, match="snapshot hash does not match"):
        plan_candidate_run(spec, broken_cases, snapshot, ["first"])


def test_read_spec_rejects_unknown_fields_and_versions(tmp_path: Path):
    payload = _payload(build_candidate_definition())
    payload["approved_labels"] = {"3": "follow"}
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid diagnostic specification"):
        read_candidate_spec(unknown)

    payload = _payload(build_candidate_definition())
    payload["schema_version"] = 2
    versioned = tmp_path / "versioned.json"
    versioned.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported diagnostic schema version"):
        read_candidate_spec(versioned)


def test_plan_rejects_unsupported_policies():
    _, cases, snapshot, _ = _load_a(policies=("first",))
    spec = read_candidate_spec(None)
    with pytest.raises(ValueError, match="unsupported policy"):
        plan_candidate_run(spec, cases, snapshot, ["llm", "other"])
