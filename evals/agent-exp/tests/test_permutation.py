"""Kiểm tra contract và replay input của permutation experiment."""

from __future__ import annotations

import pytest

import contracts
import seed_cases
from artifacts import sha256_text
from contracts import GateCase, SeedRow, SeedSnapshot, SourceFile
from permutation import build_permutation_plan, cyclic_rotations, reconstruct_trial_input
from seed_cases import prepare_cases


# Fixture giữ snapshot nhỏ để test không chạm store hoặc model thật
def _snapshot() -> SeedSnapshot:
    return SeedSnapshot(
        schema_version=1,
        snapshot_id="snapshot",
        dataset_id="dataset",
        baseline_source=SourceFile(path="baseline.json", sha256="baseline"),
        dataset_source=SourceFile(path="dataset.json", sha256="dataset"),
        whitelist_source=SourceFile(path="whitelist.json", sha256="whitelist"),
        retrieval_config={},
        unavailable_metadata={},
        internal_dieu={10, 20, 30},
        rows=[
            SeedRow(
                id=1,
                question="question",
                contexts=["Điều 10. A\nTheo Điều 20"],
                seed_dieu={10},
            )
        ],
    )


def _case(*, source_hop: int = 0, source_run_id: str | None = None, source_policy: str | None = None) -> GateCase:
    observation = "observation"
    return GateCase(
        case_id="case-1" if source_hop == 0 else "case-1-hop1",
        dataset_id="dataset",
        question_id=1,
        snapshot_id="snapshot",
        snapshot_hash="snapshot-hash",
        question="question",
        observation=observation,
        observation_hash=sha256_text(observation),
        candidates=[20],
        source_hop=source_hop,
        source_run_id=source_run_id,
        source_policy=source_policy,
        expected_action="follow",
        acceptable_dieu={20},
        label_reason="the cited article is required",
        label_status="approved",
        split="heldout",
        fewshot_overlap=False,
    )


def test_permutation_config_accepts_supported_conditions_and_defaults():
    assert hasattr(contracts, "PermutationConfig")
    config_type = contracts.PermutationConfig
    assert config_type(condition="repeat", schedule="original", repeats=3).repeats == 3
    assert config_type(condition="candidate-order", schedule="rotate", repeats=2).condition == "candidate-order"
    assert config_type(condition="seed-order", schedule="rotate", repeats=2).schedule == "rotate"
    assert config_type(condition="combined", schedule="rotate", repeats=1).condition == "combined"


@pytest.mark.parametrize(
    ("condition", "schedule"),
    [
        ("repeat", "rotate"),
        ("candidate-order", "original"),
        ("seed-order", "original"),
        ("combined", "original"),
    ],
)
def test_permutation_config_rejects_invalid_schedule(condition, schedule):
    assert hasattr(contracts, "PermutationConfig")
    with pytest.raises(ValueError, match="schedule"):
        contracts.PermutationConfig(condition=condition, schedule=schedule, repeats=1)


@pytest.mark.parametrize("repeats", [0, -1, True, False])
def test_permutation_config_rejects_nonpositive_or_boolean_repeats(repeats):
    assert hasattr(contracts, "PermutationConfig")
    with pytest.raises(ValueError, match="repeats"):
        contracts.PermutationConfig(condition="repeat", schedule="original", repeats=repeats)


def test_trial_accepts_permutation_conditions_and_rejects_unknown_condition():
    assert hasattr(contracts, "Trial")
    trial_type = contracts.Trial
    for condition in ("repeat", "candidate-order", "seed-order", "combined"):
        trial = trial_type(
            trial_id=f"trial-{condition}",
            case_id="case-1",
            repeat_id=0,
            condition=condition,
            permutation_id="p0",
            seed_order=[0],
            candidate_order=[20],
            observation_hash="observation-hash",
        )
        assert trial.condition == condition

    with pytest.raises(ValueError, match="condition"):
        trial_type(
            trial_id="trial-invalid",
            case_id="case-1",
            repeat_id=0,
            condition="random",
            permutation_id="p0",
            seed_order=[0],
            candidate_order=[20],
            observation_hash="observation-hash",
        )


def test_old_initial_selection_manifest_without_permutation_config_remains_valid():
    assert hasattr(contracts, "Trial")
    assert hasattr(contracts, "RunManifest")
    trial = contracts.Trial(
        trial_id="trial",
        case_id="case-1",
        repeat_id=0,
        condition="original",
        permutation_id="original",
        seed_order=[],
        candidate_order=[20],
        observation_hash="observation-hash",
    )
    manifest = contracts.RunManifest(
        schema_version=1,
        run_id="run",
        experiment="initial-selection",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at=None,
        status="running",
        policies=["first"],
        trials=[trial],
        cases_source=SourceFile(path="cases.jsonl", sha256="cases"),
        snapshot_source=SourceFile(path="snapshot.json", sha256="snapshot"),
        exclusions=[],
        provenance={},
        policy_config={"first": {}},
    )
    assert manifest.permutation_config is None


def test_permutation_manifest_rejects_unknown_schema_version():
    assert hasattr(contracts, "Trial")
    assert hasattr(contracts, "RunManifest")
    assert hasattr(contracts, "PermutationConfig")
    trial = contracts.Trial(
        trial_id="trial",
        case_id="case-1",
        repeat_id=0,
        condition="repeat",
        permutation_id="original",
        seed_order=[0],
        candidate_order=[20],
        observation_hash="observation-hash",
    )
    with pytest.raises(ValueError, match="schema"):
        contracts.RunManifest(
            schema_version=999,
            run_id="run",
            experiment="permutation",
            started_at="2026-01-01T00:00:00+00:00",
            ended_at=None,
            status="running",
            policies=["first"],
            trials=[trial],
            cases_source=SourceFile(path="cases.jsonl", sha256="cases"),
            snapshot_source=SourceFile(path="snapshot.json", sha256="snapshot"),
            exclusions=[],
            provenance={},
            policy_config={"first": {}},
            permutation_config=contracts.PermutationConfig(condition="repeat", schedule="original", repeats=1),
        )


def test_later_hop_case_requires_saved_source_provenance():
    assert hasattr(seed_cases, "validate_permutation_cases")
    snapshot = _snapshot()
    case = _case(source_hop=1)
    with pytest.raises(ValueError, match="source"):
        seed_cases.validate_permutation_cases(snapshot, [case], "snapshot-hash")


def test_later_hop_case_with_provenance_is_allowed_on_permutation_path():
    assert hasattr(seed_cases, "validate_permutation_cases")
    snapshot = _snapshot()
    case = _case(source_hop=1, source_run_id="run", source_policy="llm")
    selection = seed_cases.validate_permutation_cases(snapshot, [case], "snapshot-hash")
    assert selection.eligible_cases == [case]
    assert selection.exclusions == []


def test_later_hop_states_need_distinct_provenance_but_same_question_is_allowed():
    snapshot = _snapshot()
    first = _case(source_hop=1, source_run_id="run-1", source_policy="llm")
    second = first.model_copy(
        update={"case_id": "case-1-hop2", "source_hop": 2, "source_run_id": "run-2"}
    )
    selection = seed_cases.validate_permutation_cases(snapshot, [first, second], "snapshot-hash")
    assert selection.eligible_cases == [first, second]

    duplicate = second.model_copy(update={"case_id": "case-1-hop2-copy"})
    with pytest.raises(ValueError, match="duplicate later-hop state"):
        seed_cases.validate_permutation_cases(snapshot, [first, second, duplicate], "snapshot-hash")


@pytest.mark.parametrize("values", [[7], [7, 11], [7, 11, 42, 2]])
def test_cyclic_rotations_preserve_membership_and_cover_each_position(values):
    rotations = cyclic_rotations(values)
    assert len(rotations) == len(values)
    assert len({tuple(rotation) for rotation in rotations}) == len(values)
    assert rotations[0] == values
    assert all(set(rotation) == set(values) for rotation in rotations)
    for value in values:
        assert sorted(rotation.index(value) for rotation in rotations) == list(range(len(values)))


def test_build_schedule_does_not_change_when_acceptable_ids_change():
    snapshot = _snapshot()
    case = _case(source_hop=1, source_run_id="run", source_policy="llm").model_copy(
        update={"candidates": [20, 30], "acceptable_dieu": {20}}
    )
    changed = case.model_copy(update={"acceptable_dieu": {30}})
    config = contracts.PermutationConfig(
        condition="candidate-order", schedule="rotate", repeats=2
    )
    first = build_permutation_plan([case], snapshot, config)
    second = build_permutation_plan([changed], snapshot, config)
    assert [trial.model_dump(mode="json") for trial in first.trials] == [
        trial.model_dump(mode="json") for trial in second.trials
    ]


def _prepared_case(snapshot: SeedSnapshot, snapshot_hash: str) -> GateCase:
    draft = prepare_cases(snapshot, snapshot_hash=snapshot_hash)[0]
    return draft.model_copy(
        update={
            "expected_action": "follow",
            "acceptable_dieu": {next(iter(draft.candidates))},
            "label_reason": "the cited article is required",
            "label_status": "approved",
        }
    )


def _seed_snapshot(contexts: list[str], *, seed_dieu: set[int] | None = None) -> SeedSnapshot:
    snapshot = _snapshot()
    row = snapshot.rows[0].model_copy(
        update={"contexts": contexts, "seed_dieu": seed_dieu or {10}}
    )
    return snapshot.model_copy(update={"rows": [row]})


def test_candidate_order_reconstructs_rotated_candidates_with_original_observation():
    case = _case(source_hop=1, source_run_id="run", source_policy="llm").model_copy(
        update={"candidates": [20, 30]}
    )
    trial = contracts.Trial(
        trial_id="candidate-trial",
        case_id=case.case_id,
        repeat_id=0,
        condition="candidate-order",
        permutation_id="c[30,20]",
        seed_order=[],
        candidate_order=[30, 20],
        observation_hash=case.observation_hash,
    )
    reconstructed = reconstruct_trial_input(case, _snapshot(), trial)
    assert reconstructed.question == case.question
    assert reconstructed.observation == case.observation
    assert reconstructed.candidates == [30, 20]


def test_original_reconstruction_accepts_legacy_empty_seed_order():
    case = _case(source_hop=1, source_run_id="run", source_policy="llm")
    trial = contracts.Trial(
        trial_id="original-trial",
        case_id=case.case_id,
        repeat_id=0,
        condition="original",
        permutation_id="original",
        seed_order=[],
        candidate_order=list(case.candidates),
        observation_hash=case.observation_hash,
    )
    reconstructed = reconstruct_trial_input(case, _snapshot(), trial)
    assert reconstructed.model_dump() == {
        "question": case.question,
        "observation": case.observation,
        "candidates": case.candidates,
    }


def test_seed_order_rebuild_preserves_full_contexts_membership_and_candidate_set():
    snapshot = _seed_snapshot(
        [
            "Điều 10. A\nTheo Điều 20",
            "Điều 11. B\nTheo Điều 30",
            "Điều 12. C\nTheo Điều 20",
        ],
        seed_dieu={10, 11, 12},
    )
    case = _prepared_case(snapshot, "snapshot-hash")
    config = contracts.PermutationConfig(condition="seed-order", schedule="rotate", repeats=1)
    plan = build_permutation_plan([case], snapshot, config)
    assert len(plan.trials) == 3
    assert all(trial.candidate_order == case.candidates for trial in plan.trials)
    reconstructed = [reconstruct_trial_input(case, snapshot, trial) for trial in plan.trials]
    assert {20, 30} <= set(case.candidates)
    assert {input_record.candidates[0] for input_record in reconstructed} == {case.candidates[0]}
    assert all(input_record.question == case.question for input_record in reconstructed)
    assert all(
        trial.observation_hash == sha256_text(input_record.observation)
        for trial, input_record in zip(plan.trials, reconstructed)
    )


def test_seed_order_keeps_duplicate_unparseable_and_membership_only_seed_entries():
    snapshot = _seed_snapshot(
        [
            "Điều 10. A\nTheo Điều 20",
            "Điều 10. A duplicate\nTheo Điều 30",
            "No heading\nTheo Điều 40",
        ],
        seed_dieu={10, 40},
    )
    case = _prepared_case(snapshot, "snapshot-hash")
    plan = build_permutation_plan(
        [case],
        snapshot,
        contracts.PermutationConfig(condition="seed-order", schedule="rotate", repeats=1),
    )
    trial = next(item for item in plan.trials if item.seed_order == [2, 0, 1])
    reconstructed = reconstruct_trial_input(case, snapshot, trial)
    assert set(reconstructed.candidates) == set(case.candidates)
    assert "No heading" not in reconstructed.observation
    assert "Điều 10. A duplicate" in reconstructed.observation


def test_reconstruction_rejects_invalid_seed_order_and_candidate_set_drift(monkeypatch):
    snapshot = _seed_snapshot(
        ["Điều 10. A\nTheo Điều 20", "Điều 11. B\nTheo Điều 30"],
        seed_dieu={10, 11},
    )
    case = _prepared_case(snapshot, "snapshot-hash")
    invalid = contracts.Trial(
        trial_id="invalid-seed",
        case_id=case.case_id,
        repeat_id=0,
        condition="seed-order",
        permutation_id="s[0]",
        seed_order=[0],
        candidate_order=list(case.candidates),
        observation_hash=case.observation_hash,
    )
    with pytest.raises(ValueError, match="seed_order"):
        reconstruct_trial_input(case, snapshot, invalid)

    import permutation

    monkeypatch.setattr(permutation, "rebuild_case_state", lambda *_: ("changed", [999]))
    valid_shape = invalid.model_copy(update={"seed_order": [0, 1], "permutation_id": "s[0,1]"})
    with pytest.raises(ValueError, match="candidate membership"):
        reconstruct_trial_input(case, snapshot, valid_shape)


def test_seed_order_is_ineligible_for_later_hop_cases():
    snapshot = _snapshot()
    case = _case(source_hop=1, source_run_id="run", source_policy="llm")
    config = contracts.PermutationConfig(condition="seed-order", schedule="rotate", repeats=1)
    plan = build_permutation_plan([case], snapshot, config)
    assert plan.trials == []
    assert [item.reason for item in plan.exclusions] == ["later_hop_seed_order"]


def test_combined_schedule_is_an_independent_cartesian_product():
    snapshot = _seed_snapshot(
        ["Điều 10. A\nTheo Điều 20", "Điều 11. B\nTheo Điều 30"],
        seed_dieu={10, 11},
    )
    case = _prepared_case(snapshot, "snapshot-hash").model_copy(
        update={"candidates": [20, 30], "acceptable_dieu": {20}}
    )
    config = contracts.PermutationConfig(condition="combined", schedule="rotate", repeats=1)
    plan = build_permutation_plan([case], snapshot, config)
    assert len(plan.trials) == 4
    assert {(tuple(trial.seed_order), tuple(trial.candidate_order)) for trial in plan.trials} == {
        ((0, 1), (20, 30)),
        ((0, 1), (30, 20)),
        ((1, 0), (20, 30)),
        ((1, 0), (30, 20)),
    }
