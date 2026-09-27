"""Kiểm tra nhãn người duyệt, semantic split và provenance later-hop của STOP case"""

from __future__ import annotations

import json

import pytest

from diagnostic_subexp.shared.contracts import sha256_text

from stop_policy_synthetic import (
    approve,
    default_later_hop_states,
    retarget,
    reviewed_corpus,
    tamper,
)

from stop_policy_eval.cases import (
    hop0_case_id,
    import_later_hop_states,
    read_later_hop_states,
    read_stop_cases,
    validate_reviewed_cases,
    write_new_stop_cases,
)
from stop_policy_eval.contracts import LaterHopState


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory):
    """Review bundle tổng hợp dùng chung cho các test chỉ đọc"""
    return reviewed_corpus(tmp_path_factory.mktemp("reviewed"), name="shared")


def _selection(reviewed, cases):
    return validate_reviewed_cases(reviewed.snapshot, reviewed.corpus, list(cases))


def _replace(reviewed, replacement, original=None):
    """Thay case theo question/hop gốc, kể cả khi test cố tình đổi case_id hay hop"""
    target = original if original is not None else replacement
    for index, case in enumerate(reviewed.cases):
        if case.question_id == target.question_id and case.source_hop == target.source_hop:
            return [*reviewed.cases[:index], replacement, *reviewed.cases[index + 1 :]]
    raise KeyError(f"no case to replace for {target.question_id!r} at hop {target.source_hop}")


def test_valid_review_produces_eligibility_exclusions_and_readiness(reviewed):
    selection = _selection(reviewed, reviewed.cases)
    assert sorted(case.case_id for case in selection.eligible_cases) == sorted(
        case.case_id for case in reviewed.approved
    )
    reasons = {item.case_id: item.reason for item in selection.exclusions}
    assert set(reasons.values()) == {"no_candidates", "draft"}
    assert len(selection.exclusions) == 2
    assert [item.split for item in selection.readiness] == ["dev", "heldout"]
    dev = selection.readiness[0]
    assert dev.count_for("hop0_follow") == 1
    assert dev.count_for("hop0_stop") == 1
    assert dev.count_for("post_follow_follow") == 1
    assert dev.count_for("post_follow_stop") == 1
    assert dev.complete is True
    assert selection.readiness[1].complete is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("question", "Câu hỏi đã bị sửa?", "question does not match"),
        ("observation_hash", "0" * 64, "observation_hash"),
        ("snapshot_id", "seeds-deadbeefdeadbeef", "snapshot_id does not match"),
        ("snapshot_hash", "0" * 64, "snapshot hash is stale"),
        ("source_row_hash", "0" * 64, "source row hash does not match"),
        ("source_dataset_hash", "0" * 64, "source dataset hash does not match"),
        ("dataset_id", "other-dataset", "dataset_id does not match"),
        ("source_run_id", "run-x", "hop-0 cases cannot carry"),
        ("case_id", "corpus:hop0:q-deadbeef0000", "canonical hop-0 identity"),
    ],
)
def test_editing_frozen_identity_or_provenance_fails(reviewed, field, value, message):
    broken = _replace(reviewed, tamper(reviewed.case_for(1), **{field: value}))
    with pytest.raises(ValueError, match=message):
        _selection(reviewed, broken)


def test_editing_the_question_fails_for_hop_0_and_later_hops(reviewed):
    broken = _replace(
        reviewed,
        retarget(reviewed.case_for(1, source_hop=1), question="Câu hỏi khác hoàn toàn?"),
    )
    with pytest.raises(ValueError, match="question does not match"):
        _selection(reviewed, broken)


def test_editing_observation_without_refreshing_its_hash_fails(reviewed):
    broken = _replace(reviewed, tamper(reviewed.case_for(1), observation="Observation khác"))
    with pytest.raises(ValueError, match="observation_hash"):
        _selection(reviewed, broken)


def test_editing_observation_and_its_hash_fails_against_the_captured_state(reviewed):
    case = reviewed.case_for(1)
    text = f"{case.observation}\n- dòng thêm"
    broken = _replace(
        reviewed,
        tamper(
            case,
            observation=text,
            observation_hash=sha256_text(text),
            label_observation_hash=sha256_text(text),
        ),
    )
    with pytest.raises(ValueError, match="observation is stale"):
        _selection(reviewed, broken)


def test_editing_candidate_membership_or_order_fails(reviewed):
    case = reviewed.case_for(1)
    reordered = _replace(
        reviewed, tamper(case, candidates=[31, 30], acceptable_dieu=[30, 31])
    )
    with pytest.raises(ValueError, match="candidate order is stale"):
        _selection(reviewed, reordered)

    extended = _replace(
        reviewed, tamper(case, candidates=[30, 39], acceptable_dieu=[30])
    )
    with pytest.raises(ValueError, match="candidate order is stale"):
        _selection(reviewed, extended)


def test_editing_only_review_fields_is_allowed(reviewed):
    edited = tamper(
        reviewed.case_for(1),
        label_reason="Lý do đã chỉnh sửa",
        overlap_notes="Ghi chú overlap",
        fewshot_overlap=False,
    )
    selection = _selection(reviewed, _replace(reviewed, edited))
    assert edited.case_id in {item.case_id for item in selection.eligible_cases}


def test_a_stale_label_observation_hash_fails(reviewed):
    broken = _replace(
        reviewed,
        tamper(reviewed.case_for(1), label_observation_hash=sha256_text("khác")),
    )
    with pytest.raises(ValueError, match="label_observation_hash"):
        _selection(reviewed, broken)


def test_follow_labels_reject_a_wrong_or_empty_acceptable_set(reviewed):
    case = reviewed.case_for(1)
    for acceptable in ([], [31]):
        broken = _replace(reviewed, tamper(case, acceptable_dieu=acceptable))
        with pytest.raises(ValueError, match="acceptable"):
            _selection(reviewed, broken)


def test_stop_labels_reject_an_acceptable_article(reviewed):
    broken = _replace(reviewed, tamper(reviewed.case_for(2), acceptable_dieu=[31]))
    with pytest.raises(ValueError, match="acceptable"):
        _selection(reviewed, broken)


def test_approved_overlap_and_unreviewed_overlap_are_excluded_distinctly(reviewed):
    case = reviewed.case_for(1)
    overlapped = _replace(
        reviewed, tamper(case, fewshot_overlap=True, overlap_notes="trùng few-shot")
    )
    selection = _selection(reviewed, overlapped)
    reasons = {item.case_id: item.reason for item in selection.exclusions}
    assert reasons[case.case_id] == "fewshot_overlap"
    assert case.case_id not in {item.case_id for item in selection.eligible_cases}

    unreviewed = _replace(reviewed, tamper(case, fewshot_overlap=None))
    reasons = {
        item.case_id: item.reason for item in _selection(reviewed, unreviewed).exclusions
    }
    assert reasons[case.case_id] == "overlap_unreviewed"


def test_a_semantic_group_cannot_cross_splits(reviewed):
    broken = _replace(reviewed, tamper(reviewed.case_for(1), split="heldout"))
    with pytest.raises(ValueError, match="semantic group"):
        _selection(reviewed, broken)


def test_draft_group_split_assignments_cannot_cross_splits(reviewed):
    first = tamper(
        reviewed.case_for(5),
        semantic_group_id="draft-group",
        split="dev",
    )
    second = tamper(
        reviewed.case_for(6),
        semantic_group_id="draft-group",
        split="heldout",
    )
    cases = _replace(reviewed, first)
    for index, case in enumerate(cases):
        if case.case_id == second.case_id:
            cases[index] = second
            break

    with pytest.raises(ValueError, match="semantic group cannot cross"):
        _selection(reviewed, cases)


def test_one_question_cannot_split_its_hops_across_groups(reviewed):
    broken = _replace(
        reviewed,
        tamper(reviewed.case_for(1, source_hop=1), semantic_group_id="g-other-question"),
    )
    with pytest.raises(ValueError, match="same question"):
        _selection(reviewed, broken)


def test_duplicate_case_identity_or_semantic_state_fails(reviewed):
    case = reviewed.case_for(1)
    with pytest.raises(ValueError, match="duplicate case_id"):
        _selection(reviewed, [*reviewed.cases, case])

    hop1 = reviewed.case_for(1, source_hop=1)
    cloned = tamper(hop1, case_id=f"{hop1.case_id}-clone")
    with pytest.raises(ValueError, match="duplicate later-hop state"):
        _selection(reviewed, [*reviewed.cases, cloned])


def test_empty_candidate_records_stay_mechanical_exclusions(reviewed):
    attempted = approve(
        reviewed.case_for(5),
        expected_action="stop",
        acceptable_dieu=[],
        semantic_group_id="g-question-five",
        split="dev",
    )
    selection = _selection(reviewed, _replace(reviewed, attempted))
    reasons = {item.case_id: item.reason for item in selection.exclusions}
    assert reasons[attempted.case_id] == "no_candidates"
    assert attempted.case_id not in {item.case_id for item in selection.eligible_cases}


def test_hop0_roster_matches_every_typed_snapshot_identity(reviewed):
    hop0_cases = [case for case in reviewed.cases if case.source_hop == 0]
    missing = [case for case in reviewed.cases if case.case_id != hop0_cases[0].case_id]
    with pytest.raises(ValueError, match="hop-0 roster.*missing"):
        _selection(reviewed, missing)

    duplicate = tamper(hop0_cases[0], case_id=f"{hop0_cases[0].case_id}-duplicate")
    with pytest.raises(ValueError, match="hop-0 roster.*duplicate"):
        _selection(reviewed, [*reviewed.cases, duplicate])

    extra = tamper(
        hop0_cases[0],
        case_id=f"{hop0_cases[0].case_id}-typed-extra",
        question_id=str(hop0_cases[0].question_id),
    )
    with pytest.raises(ValueError, match="hop-0 roster.*extra"):
        _selection(reviewed, [*reviewed.cases, extra])


def test_empty_candidate_hop0_row_is_still_required(reviewed):
    empty_case = next(
        case for case in reviewed.cases if case.source_hop == 0 and not case.candidates
    )
    missing = [case for case in reviewed.cases if case.case_id != empty_case.case_id]

    with pytest.raises(ValueError, match="hop-0 roster.*missing"):
        _selection(reviewed, missing)


@pytest.mark.parametrize(
    "field",
    [
        "source_run_id",
        "source_policy",
        "source_trial_id",
        "parent_case_id",
        "followed_dieu",
        "source_state_hash",
    ],
)
def test_later_hop_cases_require_complete_frozen_provenance(reviewed, field):
    hop1 = reviewed.case_for(1, source_hop=1)
    broken = _replace(reviewed, tamper(hop1, **{field: None}))
    with pytest.raises(ValueError, match="later-hop cases require"):
        _selection(reviewed, broken)


def test_later_hop_parent_provenance_must_resolve(reviewed):
    hop1 = reviewed.case_for(1, source_hop=1)
    with pytest.raises(ValueError, match="parent case"):
        _selection(
            reviewed,
            _replace(reviewed, retarget(hop1, parent_case_id="synthetic:missing"), hop1),
        )
    with pytest.raises(ValueError, match="parent case"):
        _selection(reviewed, _replace(reviewed, retarget(hop1, source_hop=2), hop1))
    with pytest.raises(ValueError, match="followed article"):
        _selection(reviewed, _replace(reviewed, retarget(hop1, followed_dieu=31), hop1))


def test_later_hop_candidates_must_belong_to_the_inventory(reviewed):
    hop1 = reviewed.case_for(1, source_hop=1)
    broken = retarget(hop1, candidates=[32, 99])
    with pytest.raises(ValueError, match="whitelist"):
        _selection(reviewed, _replace(reviewed, broken, hop1))


def test_later_hop_observation_binds_to_its_explicit_state_hash(reviewed):
    hop1 = reviewed.case_for(1, source_hop=1)
    text = f"{hop1.observation}\n- dòng thêm"
    unhashed = tamper(hop1, observation=text)
    with pytest.raises(ValueError, match="observation_hash"):
        _selection(reviewed, _replace(reviewed, unhashed))

    stale = tamper(
        hop1,
        observation=text,
        observation_hash=sha256_text(text),
        label_observation_hash=sha256_text(text),
    )
    with pytest.raises(ValueError, match="source_state_hash"):
        _selection(reviewed, _replace(reviewed, stale))

    # State hợp lệ mới được chấp nhận theo hash của chính nó, không dựng lại từ hop-0
    renewed = retarget(hop1, observation=f"{hop1.observation}\n- dòng do người dùng cấp")
    selection = _selection(reviewed, _replace(reviewed, renewed, hop1))
    assert renewed.case_id in {item.case_id for item in selection.eligible_cases}
    assert renewed.case_id != hop1.case_id

    renamed = tamper(hop1, case_id="corpus:hop:q-other")
    with pytest.raises(ValueError, match="later-hop identity"):
        _selection(reviewed, _replace(reviewed, renamed))


def test_draft_and_review_artifacts_stay_usable_with_an_incomplete_matrix(tmp_path):
    incomplete = reviewed_corpus(tmp_path / "incomplete", name="incomplete")
    hop0_only = [case for case in incomplete.cases if case.source_hop == 0]
    selection = _selection(incomplete, hop0_only)
    assert selection.eligible_cases
    assert selection.readiness[0].complete is False
    assert selection.readiness[0].missing() == ["post_follow_follow", "post_follow_stop"]
    assert selection.readiness[1].missing() == ["post_follow_follow", "post_follow_stop"]
    assert read_stop_cases(incomplete.drafts_path)
    assert read_stop_cases(incomplete.reviewed_path) == incomplete.cases


def test_import_creates_a_new_version_without_touching_the_base_file(tmp_path):
    reviewed = reviewed_corpus(tmp_path / "import", name="import")
    base_cases = read_stop_cases(reviewed.drafts_path)
    base_before = reviewed.drafts_path.read_bytes()
    states = [
        LaterHopState.model_validate(payload)
        for payload in default_later_hop_states(reviewed.snapshot.dataset_id)
    ]
    states_path = tmp_path / "import" / "states.json"
    states_path.write_text(
        json.dumps(default_later_hop_states(reviewed.snapshot.dataset_id)), encoding="utf-8"
    )
    assert read_later_hop_states(states_path) == states
    states_jsonl = tmp_path / "import" / "states.jsonl"
    states_jsonl.write_text(
        "\n".join(json.dumps(payload) for payload in default_later_hop_states(reviewed.snapshot.dataset_id)),
        encoding="utf-8",
    )
    assert read_later_hop_states(states_jsonl) == states

    output = tmp_path / "import" / "second-version.jsonl"
    outcome = import_later_hop_states(reviewed.snapshot, base_cases, states, output)
    assert outcome.output_path == output
    assert len(outcome.added_case_ids) == len(states)
    assert reviewed.drafts_path.read_bytes() == base_before
    reread = read_stop_cases(output)
    assert len(reread) == len(base_cases) + len(states)
    assert {case.case_id for case in reread} == {case.case_id for case in base_cases} | set(
        outcome.added_case_ids
    )
    assert all(case.label_status == "draft" for case in reread if case.source_hop > 0)
    assert all(case.source_row_hash for case in reread if case.source_hop > 0)

    with pytest.raises(ValueError, match="destination already exists"):
        import_later_hop_states(reviewed.snapshot, base_cases, states, output)
    assert read_stop_cases(output) == reread


def test_import_rejects_states_whose_parent_case_is_absent(tmp_path):
    reviewed = reviewed_corpus(tmp_path / "orphan", name="orphan")
    base_cases = [
        case
        for case in reviewed.cases
        if case.source_hop == 0 and case.question_id != 1
    ]
    states = [
        LaterHopState.model_validate(payload)
        for payload in default_later_hop_states(reviewed.snapshot.dataset_id)
    ]
    with pytest.raises(ValueError, match="parent case"):
        import_later_hop_states(
            reviewed.snapshot, base_cases, states, tmp_path / "orphan-out.jsonl"
        )


def test_write_new_cases_refuses_to_overwrite(tmp_path):
    reviewed = reviewed_corpus(tmp_path / "writer", name="writer")
    target = tmp_path / "writer" / "existing.jsonl"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="destination already exists"):
        write_new_stop_cases(target, reviewed.cases)
    assert target.read_text(encoding="utf-8") == "{}"


def test_hop0_case_id_matches_the_canonical_formula(tmp_path):
    reviewed = reviewed_corpus(tmp_path / "identity", name="identity")
    for case in reviewed.cases:
        if case.source_hop == 0:
            assert case.case_id == hop0_case_id(reviewed.snapshot.dataset_id, case.question_id)
