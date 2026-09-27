"""Kiểm tra contract STOP-local và các invariant liên field của nó."""

from __future__ import annotations

import copy

import pytest

from metrics import ratio_metric
from diagnostic_subexp.shared.contracts import sha256_text, structured_hash
from stop_policy_eval.contracts import (
    BASELINE_VARIANT,
    REQUIREMENT_CLASSES,
    STOP_POLICY_SCHEMA_VERSION,
    LaterHopState,
    StopLabelRates,
    StopPolicyCase,
    StopReadiness,
    StopRequirementCount,
    StopTrial,
    frozen_state_payload,
    later_hop_state_hash,
)

OBSERVATION = "Các Điều đã thu thập:\n[Context 30] Điều 30. Ví dụ\n- Xem Điều 31 Quy chế này."
QUESTION = "Câu hỏi ví dụ về điều kiện tốt nghiệp?"


def _row_hash(value: str = "row") -> str:
    return sha256_text(value)


def _case_payload(**overrides) -> dict:
    """Dựng payload case hợp lệ rồi phủ các field cần phá trong từng test."""
    payload = {
        "case_id": "synthetic:hop0:q-000000000001",
        "dataset_id": "synthetic",
        "question_id": 1,
        "snapshot_id": "seeds-0123456789abcdef",
        "snapshot_hash": "a" * 64,
        "question": QUESTION,
        "observation": OBSERVATION,
        "observation_hash": sha256_text(OBSERVATION),
        "candidates": [30, 31],
        "source_hop": 0,
        "source_run_id": None,
        "source_policy": None,
        "source_trial_id": None,
        "parent_case_id": None,
        "followed_dieu": None,
        "source_state_hash": None,
        "followed_article_sha256": None,
        "collected_article_refs": [],
        "expected_action": "follow",
        "acceptable_dieu": [31],
        "label_reason": "Điều 31 là dependency còn thiếu",
        "label_status": "approved",
        "label_observation_hash": sha256_text(OBSERVATION),
        "semantic_group_id": "group-alpha",
        "split": "dev",
        "fewshot_overlap": False,
        "overlap_notes": None,
        "source_row_hash": _row_hash(),
        "source_dataset_hash": "b" * 64,
    }
    payload.update(overrides)
    return payload


def _case(**overrides) -> StopPolicyCase:
    return StopPolicyCase.model_validate(_case_payload(**overrides))


def _draft_payload(**overrides) -> dict:
    payload = _case_payload(
        expected_action="unresolved",
        acceptable_dieu=[],
        label_reason=None,
        label_status="draft",
        label_observation_hash=None,
        semantic_group_id=None,
        split="unassigned",
        fewshot_overlap=None,
    )
    payload.update(overrides)
    return payload


def _later_hop_payload(**overrides) -> dict:
    payload = {
        "dataset_id": "synthetic",
        "question_id": 1,
        "source_hop": 1,
        "question": QUESTION,
        "observation": "Các Điều đã thu thập:\n[Context 31] Điều 31. Ví dụ",
        "candidates": [32, 33],
        "source_run_id": "run-abc",
        "source_policy": "llm",
        "source_trial_id": "synthetic:hop0:q-000000000001:r0",
        "parent_case_id": "synthetic:hop0:q-000000000001",
        "followed_dieu": 31,
        "followed_article_sha256": "a" * 64,
        "collected_article_refs": [{"dieu": 31, "sha256": "a" * 64}],
    }
    payload.update(overrides)
    return payload


def _hop_state(**overrides) -> LaterHopState:
    return LaterHopState.model_validate(_later_hop_payload(**overrides))


def _hop_case_payload(**overrides) -> dict:
    state = _hop_state()
    payload = _case_payload(
        case_id="synthetic:hop1:q-000000000001.c0ffee01",
        observation=state.observation,
        observation_hash=sha256_text(state.observation),
        label_observation_hash=sha256_text(state.observation),
        candidates=list(state.candidates),
        source_hop=state.source_hop,
        source_run_id=state.source_run_id,
        source_policy=state.source_policy,
        source_trial_id=state.source_trial_id,
        parent_case_id=state.parent_case_id,
        followed_dieu=state.followed_dieu,
        followed_article_sha256=state.followed_article_sha256,
        collected_article_refs=[
            item.model_dump(mode="json") for item in state.collected_article_refs
        ],
        source_state_hash=state.state_hash,
        acceptable_dieu=[state.candidates[0]],
    )
    payload.update(overrides)
    return payload


def _hop_case(**overrides) -> StopPolicyCase:
    return StopPolicyCase.model_validate(_hop_case_payload(**overrides))


def test_unknown_fields_are_rejected():
    with pytest.raises(ValueError, match="Extra inputs"):
        _case(unapproved_shortcut=True)
    with pytest.raises(ValueError, match="Extra inputs"):
        LaterHopState.model_validate({**_later_hop_payload(), "label": "stop"})


def test_question_identity_is_typed_not_coerced():
    assert _case(question_id="1").question_id == "1"
    assert _case(question_id="1").question_id != 1
    with pytest.raises(ValueError, match="integer or string"):
        _case(question_id=True)
    with pytest.raises(ValueError, match="integer or string"):
        _case(question_id=1.0)


def test_candidates_must_be_positive_and_unique():
    with pytest.raises(ValueError, match="invalid article id"):
        _case(candidates=[30, 0], acceptable_dieu=[])
    with pytest.raises(ValueError, match="invalid article id"):
        _case(candidates=[30, -1], acceptable_dieu=[])
    with pytest.raises(ValueError, match="invalid article id|valid integer"):
        _case(candidates=[30, True], acceptable_dieu=[])
    with pytest.raises(ValueError, match="duplicate article id"):
        _case(candidates=[30, 30], acceptable_dieu=[30])


def test_draft_cases_default_to_unassigned_unreviewed_state():
    case = StopPolicyCase.model_validate(_draft_payload())
    assert case.expected_action == "unresolved"
    assert case.label_status == "draft"
    assert case.split == "unassigned"
    assert case.semantic_group_id is None
    assert case.fewshot_overlap is None
    assert case.label_observation_hash is None
    assert case.acceptable_dieu == []


def test_draft_cases_can_keep_approved_split_and_group_before_label_approval():
    assigned = StopPolicyCase.model_validate(
        _draft_payload(split="dev", semantic_group_id="group-alpha")
    )
    assert assigned.label_status == "draft"
    assert assigned.split == "dev"
    assert assigned.semantic_group_id == "group-alpha"
    with pytest.raises(ValueError, match="draft cases cannot carry a label observation hash"):
        StopPolicyCase.model_validate(
            _draft_payload(label_observation_hash=sha256_text(OBSERVATION))
        )


def test_approved_cases_require_a_nonblank_rationale():
    for reason in (None, "", "   "):
        with pytest.raises(ValueError, match="non-blank label_reason"):
            _case(label_reason=reason)


def test_approval_binds_the_exact_observation_hash():
    with pytest.raises(ValueError, match="label_observation_hash"):
        _case(label_observation_hash=sha256_text(f"{OBSERVATION} khác"))
    with pytest.raises(ValueError, match="label_observation_hash"):
        _case(label_observation_hash=None)


def test_stored_observation_hash_must_match_its_bytes():
    with pytest.raises(ValueError, match="observation_hash"):
        _case(observation_hash="c" * 64, label_observation_hash="c" * 64)


def test_follow_label_requires_a_nonempty_acceptable_subset():
    with pytest.raises(ValueError, match="follow labels require"):
        _case(acceptable_dieu=[])
    with pytest.raises(ValueError, match="subset of candidates"):
        _case(acceptable_dieu=[31, 99])
    assert _case(acceptable_dieu=[30, 31]).acceptable_dieu == [30, 31]


def test_stop_label_requires_an_empty_acceptable_set():
    with pytest.raises(ValueError, match="stop labels require empty acceptable_dieu"):
        _case(expected_action="stop", acceptable_dieu=[31])
    stop = _case(expected_action="stop", acceptable_dieu=[])
    assert stop.acceptable_dieu == []


def test_empty_candidates_stay_mechanical_not_positive_stop():
    # Reviewer gán STOP cho frontier rỗng vẫn load được để validator loại cơ học
    attempted = _case(expected_action="stop", candidates=[], acceptable_dieu=[])
    assert attempted.candidates == [] and attempted.expected_action == "stop"
    with pytest.raises(ValueError, match="no_candidates cases must have an empty candidate list"):
        _case(expected_action="no_candidates", candidates=[30], acceptable_dieu=[])
    with pytest.raises(ValueError, match="follow labels require candidates"):
        _case(expected_action="follow", candidates=[], acceptable_dieu=[])


def test_split_assignment_is_enforced():
    with pytest.raises(ValueError, match="approved cases require dev or heldout"):
        _case(split="unassigned")
    with pytest.raises(ValueError, match="Input should be"):
        _case(split="test")


def test_fewshot_overlap_requires_a_decided_boolean():
    for value in (0, 1, "false"):
        with pytest.raises(ValueError):
            _case(fewshot_overlap=value)
    assert _case(fewshot_overlap=True, overlap_notes="trùng few-shot hiện tại").fewshot_overlap is True
    assert _case(fewshot_overlap=None).fewshot_overlap is None


def test_approved_cases_require_a_nonempty_semantic_group():
    with pytest.raises(ValueError, match="approved cases require a semantic_group_id"):
        _case(semantic_group_id=None)
    with pytest.raises(ValueError, match="semantic_group_id"):
        _case(semantic_group_id="   ")


def test_later_hop_provenance_must_be_complete():
    for field in (
        "source_run_id",
        "source_policy",
        "source_trial_id",
        "parent_case_id",
        "followed_dieu",
        "followed_article_sha256",
    ):
        with pytest.raises(ValueError, match="later-hop cases require"):
            _hop_case(**{field: None})
    with pytest.raises(ValueError, match="later-hop cases require"):
        _hop_case(source_state_hash=None)
    with pytest.raises(ValueError):
        LaterHopState.model_validate(_later_hop_payload(source_run_id=None))
    with pytest.raises(ValueError, match="positive hop"):
        LaterHopState.model_validate(_later_hop_payload(source_hop=0))
    with pytest.raises(ValueError, match="must not be empty"):
        LaterHopState.model_validate(_later_hop_payload(candidates=[]))


def test_hop_zero_cases_cannot_carry_later_hop_provenance():
    with pytest.raises(ValueError, match="hop-0 cases cannot carry"):
        _case(source_run_id="run-abc")
    with pytest.raises(ValueError, match="hop-0 cases cannot carry"):
        _case(source_state_hash="d" * 64)


def test_frozen_state_hash_binds_question_observation_and_provenance():
    state = LaterHopState.model_validate(_later_hop_payload())
    assert state.state_hash == structured_hash(state.payload())
    payload = frozen_state_payload(
        dataset_id=state.dataset_id,
        question_id=state.question_id,
        source_hop=state.source_hop,
        question=state.question,
        observation=state.observation,
        candidates=list(state.candidates),
        source_run_id=state.source_run_id,
        source_policy=state.source_policy,
        source_trial_id=state.source_trial_id,
        parent_case_id=state.parent_case_id,
        followed_dieu=state.followed_dieu,
        followed_article_sha256=state.followed_article_sha256,
        collected_article_refs=[
            item.model_dump(mode="json") for item in state.collected_article_refs
        ],
    )
    assert structured_hash(payload) == state.state_hash
    edited = LaterHopState.model_validate(
        _later_hop_payload(observation=f"{state.observation}\n- thêm dòng")
    )
    assert edited.state_hash != state.state_hash
    changed_article = LaterHopState.model_validate(
        _later_hop_payload(
            followed_article_sha256="b" * 64,
            collected_article_refs=[{"dieu": 31, "sha256": "b" * 64}],
        )
    )
    assert changed_article.state_hash != state.state_hash
    case = _hop_case()
    assert later_hop_state_hash(case) == state.state_hash


def test_hop_case_state_hash_must_match_the_stored_fields():
    with pytest.raises(ValueError, match="source_state_hash does not match"):
        _hop_case(source_state_hash="e" * 64)
    with pytest.raises(ValueError, match="later-hop cases require"):
        _hop_case(followed_dieu=None)


def test_readiness_matrix_covers_four_requirement_classes():
    assert REQUIREMENT_CLASSES == (
        "hop0_follow",
        "hop0_stop",
        "post_follow_follow",
        "post_follow_stop",
    )
    counts = {
        "hop0_follow": 2,
        "hop0_stop": 0,
        "post_follow_follow": 1,
        "post_follow_stop": 1,
    }
    rows = [
        StopRequirementCount(
            requirement=name,
            eligible_cases=counts[name],
            satisfied=counts[name] > 0,
        )
        for name in REQUIREMENT_CLASSES
    ]
    readiness = StopReadiness(split="dev", requirements=rows, complete=False)
    assert [row.requirement for row in readiness.requirements] == list(REQUIREMENT_CLASSES)
    assert readiness.missing() == ["hop0_stop"]
    with pytest.raises(ValueError, match="exactly the four requirement classes"):
        StopReadiness(split="dev", requirements=rows[:3], complete=False)
    with pytest.raises(ValueError, match="complete must match"):
        StopReadiness(split="dev", requirements=rows, complete=True)
    with pytest.raises(ValueError, match="satisfied must match"):
        StopReadiness(
            split="dev",
            requirements=[
                *rows[:1],
                StopRequirementCount(requirement="hop0_stop", eligible_cases=1, satisfied=False),
                *rows[2:],
            ],
            complete=False,
        )


def test_zero_denominator_ratios_are_null_and_counts_stay_consistent():
    rates = StopLabelRates(
        follow_accuracy=ratio_metric(1, 2),
        false_stop_rate=ratio_metric(1, 2),
        stop_accuracy=ratio_metric(0, 0),
        over_hop_rate=ratio_metric(0, 0),
        follow_valid=2,
        stop_valid=0,
    )
    assert rates.stop_accuracy.value is None and rates.over_hop_rate.value is None
    with pytest.raises(ValueError, match="follow_valid"):
        StopLabelRates(
            follow_accuracy=ratio_metric(1, 2),
            false_stop_rate=ratio_metric(1, 2),
            stop_accuracy=ratio_metric(0, 0),
            over_hop_rate=ratio_metric(0, 0),
            follow_valid=3,
            stop_valid=0,
        )
    with pytest.raises(ValueError, match="complementary counts"):
        StopLabelRates(
            follow_accuracy=ratio_metric(1, 2),
            false_stop_rate=ratio_metric(0, 2),
            stop_accuracy=ratio_metric(0, 0),
            over_hop_rate=ratio_metric(0, 0),
            follow_valid=2,
            stop_valid=0,
        )


def test_trial_identity_is_unique_and_label_free():
    trial = StopTrial(
        trial_id="synthetic:hop0:q-000000000001:r0",
        input_trace_id="synthetic:hop0:q-000000000001:baseline",
        case_id="synthetic:hop0:q-000000000001",
        variant=BASELINE_VARIANT,
        repeat_id=0,
        input_fingerprint="f" * 64,
    )
    assert trial.repeat_id == 0
    with pytest.raises(ValueError, match="Extra inputs"):
        StopTrial(
            **trial.model_dump(),
            expected_action="stop",
        )


def test_schema_version_constant_is_exported_from_the_package():
    import stop_policy_eval

    assert stop_policy_eval.STOP_POLICY_SCHEMA_VERSION == STOP_POLICY_SCHEMA_VERSION == 1
    assert stop_policy_eval.__all__ == ["STOP_POLICY_SCHEMA_VERSION"]


def test_case_payload_round_trips_through_json():
    case = _case()
    payload = copy.deepcopy(case.model_dump(mode="json"))
    assert StopPolicyCase.model_validate(payload) == case
    assert payload["acceptable_dieu"] == [31]
