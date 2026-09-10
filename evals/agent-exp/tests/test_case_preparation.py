"""Offline parity and label-boundary checks for prepared gate cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifacts import sha256_file, sha256_text, write_snapshot
from contracts import GateCase
from seed_cases import (
    import_seeds,
    normalize_seed,
    policy_input,
    prepare_cases,
    validate_replay,
)
from src.rag.agent.loop import agent_retrieve
from src.rag.agent.tools import extract_citation_mentions


def _snapshot(tmp_path: Path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps([{"id": 1, "user_input": "Câu hỏi"}], ensure_ascii=False),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "config": {"agent": False},
                "results": [
                    {
                        "id": 1,
                        "query": "Câu hỏi",
                        "retrieved_contexts": [
                            "Điều 10. A\nTheo Điều 20 và Điều 30",
                            "Điều 10. B\nTheo Điều 40",
                            "Không có heading\nTheo Điều 50",
                        ],
                        "hop_scores": {"retrieved_dieu": [10, 40]},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    whitelist = tmp_path / "whitelist.json"
    whitelist.write_text("[10, 20, 30, 40, 50]", encoding="utf-8")
    snapshot = import_seeds(baseline, dataset, whitelist)
    snapshot_path = tmp_path / "snapshot.json"
    write_snapshot(snapshot, snapshot_path)
    return snapshot, snapshot_path


def _validated_case(case: GateCase, **updates) -> GateCase:
    payload = case.model_dump(mode="python")
    payload.update(updates)
    return GateCase.model_validate(payload)


def test_normalize_seed_preserves_duplicates_unrecognized_text_and_membership(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    row = snapshot.rows[0]
    normalized = normalize_seed(row)

    # Seed trùng hoặc không nhận diện được heading vẫn phải giữ nguyên text và thứ tự
    assert list(normalized.values()) == row.contexts + [None]
    assert list(normalized)[:3] == [10, -2, -3]
    # ID chỉ có trong membership nhận None, không được bịa ánh xạ sang context
    assert normalized[40] is None


def test_prepare_cases_reuses_canonical_frontier_and_observation(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    cases = prepare_cases(snapshot, snapshot_hash=sha256_file(snapshot_path))

    assert len(cases) == 1
    case = cases[0]
    assert case.candidates == [20, 30, 50]
    assert case.expected_action == "unresolved"
    assert case.label_status == "draft"
    assert case.acceptable_dieu == set()
    assert case.source_hop == 0
    assert case.source_run_id is None
    assert case.source_policy is None
    assert case.split == "dev"
    assert case.fewshot_overlap is None
    assert case.observation_hash == sha256_text(case.observation)
    assert "Theo Điều 20 và Điều 30" in case.observation
    assert "Không có heading" not in case.observation
    assert policy_input(case).model_dump() == {
        "question": "Câu hỏi",
        "observation": case.observation,
        "candidates": [20, 30, 50],
    }


def test_empty_frontier_is_a_mechanical_exclusion(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    row = snapshot.rows[0]
    row = row.model_copy(update={"contexts": ["Điều 10. A"], "seed_dieu": {10}})
    snapshot = snapshot.model_copy(update={"rows": [row]})
    cases = prepare_cases(snapshot, snapshot_hash=sha256_file(snapshot_path))
    # Frontier rỗng là loại cơ học, không phải một quyết định STOP của model
    assert cases[0].expected_action == "no_candidates"


def test_policy_input_rejects_empty_candidates(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    row = snapshot.rows[0].model_copy(update={"contexts": ["Điều 10. A"], "seed_dieu": {10}})
    empty_snapshot = snapshot.model_copy(update={"rows": [row]})
    case = prepare_cases(empty_snapshot, snapshot_hash=sha256_file(snapshot_path))[0]
    assert case.candidates == []

    with pytest.raises(ValueError, match="non-empty candidates"):
        policy_input(case)


def test_approved_follow_requires_candidate_subset_and_rationale(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    snapshot_hash = sha256_file(snapshot_path)
    draft = prepare_cases(snapshot, snapshot_hash=snapshot_hash)[0]
    approved = _validated_case(
        draft,
        expected_action="follow",
        acceptable_dieu=[20],
        label_reason="Điều 20 trả lời phần được hỏi",
        label_status="approved",
    )
    selected = validate_replay(snapshot, [approved], snapshot_hash=snapshot_hash)
    assert selected.eligible_cases == [approved]
    assert selected.exclusions == []


def test_stale_observation_is_rejected_before_policy_calls(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    snapshot_hash = sha256_file(snapshot_path)
    case = prepare_cases(snapshot, snapshot_hash=snapshot_hash)[0]
    stale = _validated_case(case, observation="stale")

    with pytest.raises(ValueError, match="observation"):
        validate_replay(snapshot, [stale], snapshot_hash=snapshot_hash)


def test_stale_snapshot_hash_is_rejected_before_policy_calls(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    cases = prepare_cases(snapshot, snapshot_hash=sha256_file(snapshot_path))
    with pytest.raises(ValueError, match="snapshot hash is stale"):
        validate_replay(snapshot, cases, snapshot_hash="stale-snapshot")


def test_duplicate_question_under_a_forged_case_id_is_rejected(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    snapshot_hash = sha256_file(snapshot_path)
    case = prepare_cases(snapshot, snapshot_hash=snapshot_hash)[0]
    duplicate = _validated_case(case, case_id="dataset:hop0:q-forged")

    with pytest.raises(ValueError, match="q-forged"):
        validate_replay(snapshot, [duplicate, case], snapshot_hash=snapshot_hash)


def test_swapped_case_identity_is_rejected_by_canonical_id(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    snapshot_hash = sha256_file(snapshot_path)
    second_row = snapshot.rows[0].model_copy(update={"id": 2, "question": "Câu hỏi hai"})
    grown = snapshot.model_copy(update={"rows": [*snapshot.rows, second_row]})
    cases = prepare_cases(grown, snapshot_hash=snapshot_hash)
    swapped = [
        _validated_case(cases[0], case_id=cases[1].case_id),
        _validated_case(cases[1], case_id=cases[0].case_id),
    ]

    with pytest.raises(ValueError, match="canonical"):
        validate_replay(grown, swapped, snapshot_hash=snapshot_hash)


def test_case_file_missing_a_snapshot_row_is_rejected(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    snapshot_hash = sha256_file(snapshot_path)
    second_row = snapshot.rows[0].model_copy(update={"id": 2, "question": "Câu hỏi hai"})
    grown = snapshot.model_copy(update={"rows": [*snapshot.rows, second_row]})
    cases = prepare_cases(grown, snapshot_hash=snapshot_hash)
    assert len(cases) == 2

    with pytest.raises(ValueError, match="missing"):
        validate_replay(grown, [cases[0]], snapshot_hash=snapshot_hash)


def test_empty_case_file_is_rejected_as_incomplete_roster(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    snapshot_hash = sha256_file(snapshot_path)

    with pytest.raises(ValueError, match="missing"):
        validate_replay(snapshot, [], snapshot_hash=snapshot_hash)


def test_follow_label_outside_candidates_is_rejected(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    snapshot_hash = sha256_file(snapshot_path)
    case = prepare_cases(snapshot, snapshot_hash=snapshot_hash)[0]
    with pytest.raises(ValueError, match="subset"):
        _validated_case(
            case,
            expected_action="follow",
            acceptable_dieu=[999],
            label_reason="Không hợp lệ",
            label_status="approved",
        )


def test_stale_candidate_order_is_rejected_before_policy_calls(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    snapshot_hash = sha256_file(snapshot_path)
    case = prepare_cases(snapshot, snapshot_hash=snapshot_hash)[0]
    stale = _validated_case(case, candidates=[999])
    with pytest.raises(ValueError, match="candidate order is stale"):
        validate_replay(snapshot, [stale], snapshot_hash=snapshot_hash)


def test_approved_label_invariants_reject_invalid_follow_and_stop_labels(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    snapshot_hash = sha256_file(snapshot_path)
    case = prepare_cases(snapshot, snapshot_hash=snapshot_hash)[0]
    with pytest.raises(ValueError, match="follow labels require"):
        _validated_case(case, expected_action="follow", acceptable_dieu=[], label_reason="reason", label_status="approved")
    with pytest.raises(ValueError, match="stop labels require"):
        _validated_case(case, expected_action="stop", acceptable_dieu=[20], label_reason="reason", label_status="approved")
    with pytest.raises(ValueError, match="approved semantic labels"):
        _validated_case(case, expected_action="stop", acceptable_dieu=[], label_reason="   ", label_status="approved")


class _StopAfterGate(Exception):
    """Abort parity execution after capturing the initial gate input."""


def test_prepare_cases_matches_real_loop_boundary_before_follow(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    cases = prepare_cases(snapshot, snapshot_hash=sha256_file(snapshot_path))
    captured = []
    fetched = []

    def capture_decide(question, observation, candidates):
        captured.append((question, observation, list(candidates)))
        raise _StopAfterGate

    def fail_fetch(dieu):
        fetched.append(dieu)
        raise AssertionError("fetch must not run before the gate returns")

    row = snapshot.rows[0]
    with pytest.raises(_StopAfterGate):
        agent_retrieve(
            row.question,
            retrieve_seed_fn=lambda question: (list(row.contexts), set(row.seed_dieu)),
            get_article_fn=fail_fetch,
            extract_citations_fn=extract_citation_mentions,
            decide_fn=capture_decide,
            internal_dieu=snapshot.internal_dieu,
            max_follow_articles=3,
        )

    # Chạy tới ranh giới gate của loop thật rồi dừng trước khi follow hoặc fetch
    case = next(item for item in cases if item.candidates)
    gate = policy_input(case)
    assert captured == [(gate.question, gate.observation, gate.candidates)]
    assert fetched == []


def test_empty_frontier_makes_no_gate_or_fetch_calls(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    row = snapshot.rows[0].model_copy(update={"contexts": ["Điều 10. A"], "seed_dieu": {10}})
    snapshot = snapshot.model_copy(update={"rows": [row]})
    calls = {"gate": 0, "fetch": 0}

    def decide(*args):
        calls["gate"] += 1
        raise AssertionError("empty frontier must not call gate")

    def fetch(*args):
        calls["fetch"] += 1
        raise AssertionError("empty frontier must not fetch")

    agent_retrieve(
        row.question,
        retrieve_seed_fn=lambda question: (list(row.contexts), set(row.seed_dieu)),
        get_article_fn=fetch,
        extract_citations_fn=extract_citation_mentions,
        decide_fn=decide,
        internal_dieu=snapshot.internal_dieu,
        max_follow_articles=3,
    )
    assert calls == {"gate": 0, "fetch": 0}


def test_replay_exclusions_keep_mixed_roster_and_approved_case(tmp_path):
    snapshot, snapshot_path = _snapshot(tmp_path)
    second_row = snapshot.rows[0].model_copy(update={"id": 2, "question": "Câu hỏi hai"})
    snapshot = snapshot.model_copy(update={"rows": [snapshot.rows[0], second_row]})
    snapshot_hash = sha256_file(snapshot_path)
    cases = prepare_cases(snapshot, snapshot_hash=snapshot_hash)
    cases[0] = _validated_case(
        cases[0],
        expected_action="stop",
        acceptable_dieu=[],
        label_reason="Đã duyệt dừng",
        label_status="approved",
    )
    cases[1] = _validated_case(
        cases[1],
        expected_action="follow",
        acceptable_dieu=[20],
        label_reason="Chờ duyệt",
        label_status="draft",
    )
    selection = validate_replay(snapshot, cases, snapshot_hash=snapshot_hash)
    assert [case.case_id for case in selection.eligible_cases] == [cases[0].case_id]
    assert {item.reason for item in selection.exclusions} == {"draft"}


def test_policy_input_contains_only_label_free_fields(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    case = prepare_cases(snapshot, snapshot_hash="snapshot-hash")[0]
    gate_input = policy_input(case)
    # Policy chỉ nhận question, observation và candidates, không nhận nhãn hay lý do chấm
    assert set(gate_input.model_dump()) == {"question", "observation", "candidates"}
    assert "expected_action" not in gate_input.model_dump()
    assert "label_reason" not in gate_input.model_dump()
