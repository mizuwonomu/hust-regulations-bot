"""Fixture tổng hợp dùng chung cho test STOP-policy, không cần corpus hay model thật

Mọi định danh, câu hỏi và số Điều ở đây chỉ tồn tại trong test; không fixture nào
được sao chép từ corpus người dùng
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.common.single_pass_retrieval import SinglePassSettings

from artifacts import read_snapshot
from contracts import SeedSnapshot
from stop_policy_eval.capture import StopCaptureOutcome, capture_corpus
from stop_policy_eval.cases import (
    StopCorpus,
    hop0_case_id,
    import_later_hop_states,
    later_hop_case_id,
    read_stop_corpus,
    validate_reviewed_cases,
    write_new_stop_cases,
)
from stop_policy_eval.contracts import LaterHopState, StopPolicyCase
from diagnostic_subexp.shared.contracts import sha256_text

CORPUS_ANSWER_MARKER = "CORPUS_ANSWER_ONLY_MARKER"
CORPUS_GOLD_MARKER = "CORPUS_GOLD_CONTEXT_MARKER"
CORPUS_LINK_MARKER = "CORPUS_LINK_ONLY_MARKER"

WHITELIST = [21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35]

QUESTION_ONE = "Điều kiện xét tốt nghiệp kỹ sư gồm những gì?"
QUESTION_TWO = "Điều kiện đánh giá đồ án tốt nghiệp kỹ sư?"
QUESTION_THREE = "Quy định học phí học kỳ chính?"
QUESTION_FOUR = "Cách xếp hạng tốt nghiệp kỹ sư?"
QUESTION_FIVE = "Thủ tục rút học phần đã đăng ký?"
QUESTION_SIX = "Điểm học phần tối thiểu để đạt?"

CONTEXT_21 = "Điều 21. Điều kiện tốt nghiệp kỹ sư\nQuy trình xét tốt nghiệp theo Điều 30 Quy chế này."
CONTEXT_22 = "Điều 22. Đồ án tốt nghiệp kỹ sư\nĐiều kiện đánh giá đồ án theo Điều 31 Quy chế này."
CONTEXT_23 = "Điều 23. Học phí học kỳ chính\nHọc phí thực hiện theo Điều 33 Quy chế này."
CONTEXT_24 = "Điều 24. Xếp hạng tốt nghiệp\nHạng tốt nghiệp theo Điều 35 Quy chế này."
CONTEXT_25 = "Điều 25. Rút học phần\nThủ tục rút học phần theo Điều 9 Quy chế này."
CONTEXT_26 = "Điều 26. Điểm học phần\nĐiểm đạt tối thiểu theo Điều 34 Quy chế này."

HOP1_OBSERVATION_ONE = (
    "Các Điều đã thu thập:\n"
    "[Context 21] Điều 21. Điều kiện tốt nghiệp kỹ sư\n"
    "- Quy trình xét tốt nghiệp theo Điều 30 Quy chế này.\n"
    "[Context 30] Điều 30. Quy trình xét tốt nghiệp\n"
    "- Hồ sơ xét tốt nghiệp theo Điều 32 Quy chế này."
)
HOP1_OBSERVATION_TWO = (
    "Các Điều đã thu thập:\n"
    "[Context 22] Điều 22. Đồ án tốt nghiệp kỹ sư\n"
    "- Điều kiện đánh giá đồ án theo Điều 31 Quy chế này.\n"
    "[Context 31] Điều 31. Đánh giá đồ án\n"
    "- Hội đồng đánh giá theo Điều 31 Quy chế này."
)
HOP1_OBSERVATION_THREE = (
    "Các Điều đã thu thập:\n"
    "[Context 23] Điều 23. Học phí học kỳ chính\n"
    "- Học phí thực hiện theo Điều 33 Quy chế này.\n"
    "[Context 33] Điều 33. Miễn giảm học phí\n"
    "- Đối tượng miễn giảm theo Điều 34 Quy chế này."
)
HOP1_OBSERVATION_FOUR = (
    "Các Điều đã thu thập:\n"
    "[Context 24] Điều 24. Xếp hạng tốt nghiệp\n"
    "- Hạng tốt nghiệp theo Điều 35 Quy chế này.\n"
    "[Context 35] Điều 35. Bảng điểm tốt nghiệp\n"
    "- Bảng điểm do phòng đào tạo cấp."
)


def default_corpus_rows() -> list[dict[str, Any]]:
    """Dựng corpus tổng hợp sáu câu, đủ bốn lớp nhãn và hai case loại cơ học"""
    return [
        {
            "id": 1,
            "user_input": QUESTION_ONE,
            "response": f"{CORPUS_ANSWER_MARKER} một",
            "retrieved_contexts": f"{CORPUS_GOLD_MARKER} một",
            "type": "multi_hop",
            "group": "Treatment",
            "link": f"{CORPUS_LINK_MARKER} 21 -> 30",
        },
        {
            "id": 2,
            "user_input": QUESTION_TWO,
            "response": f"{CORPUS_ANSWER_MARKER} hai",
            "retrieved_contexts": f"{CORPUS_GOLD_MARKER} hai",
            "type": "single",
            "group": "Control",
            "link": f"{CORPUS_LINK_MARKER} 22 -> 31",
        },
        {
            "id": 3,
            "user_input": QUESTION_THREE,
            "response": f"{CORPUS_ANSWER_MARKER} ba",
            "retrieved_contexts": f"{CORPUS_GOLD_MARKER} ba",
            "type": "multi_hop",
            "group": "Treatment",
            "link": f"{CORPUS_LINK_MARKER} 23 -> 33",
        },
        {
            "id": 4,
            "user_input": QUESTION_FOUR,
            "response": f"{CORPUS_ANSWER_MARKER} bốn",
            "retrieved_contexts": f"{CORPUS_GOLD_MARKER} bốn",
            "type": "single",
            "group": "Control",
            "link": f"{CORPUS_LINK_MARKER} 24 -> 35",
        },
        {
            "id": 5,
            "user_input": QUESTION_FIVE,
            "response": f"{CORPUS_ANSWER_MARKER} năm",
            "retrieved_contexts": f"{CORPUS_GOLD_MARKER} năm",
            "type": "single",
            "group": "Control",
            "link": f"{CORPUS_LINK_MARKER} 25 -> 9",
        },
        {
            "id": 6,
            "user_input": QUESTION_SIX,
            "response": f"{CORPUS_ANSWER_MARKER} sáu",
            "retrieved_contexts": f"{CORPUS_GOLD_MARKER} sáu",
            "type": "single",
            "group": "Control",
            "link": f"{CORPUS_LINK_MARKER} 26 -> 34",
        },
    ]


def default_answers() -> dict[str, tuple[list[str], set[int]]]:
    """Dựng frontier hop-0 tổng hợp: bốn câu có candidate, một câu rỗng, một câu rỗng"""
    return {
        QUESTION_ONE: ([CONTEXT_21], {21}),
        QUESTION_TWO: ([CONTEXT_22], {22}),
        QUESTION_THREE: ([CONTEXT_23], {23}),
        QUESTION_FOUR: ([CONTEXT_24], {24}),
        QUESTION_FIVE: ([CONTEXT_25], {25}),
        QUESTION_SIX: ([CONTEXT_26], {26}),
    }


class FakeRuntime:
    """Runtime retrieval giả ghi lại câu hỏi nhận được"""

    def __init__(self, answers: dict[str, tuple[list[str], set[int]]], *, error: Exception | None = None):
        self.answers = answers
        self.error = error
        self.questions: list[str] = []
        self.metadata = {"retrieval": "fake"}

    def retrieve(self, question: str) -> tuple[list[str], set[int]]:
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        if question not in self.answers:
            raise KeyError(f"unexpected question: {question}")
        return self.answers[question]


class FakeFactory:
    """Factory runtime giả đếm số lần dựng runtime"""

    def __init__(self, runtime: FakeRuntime):
        self.runtime = runtime
        self.calls = 0

    def __call__(self, settings: Any) -> FakeRuntime:
        self.calls += 1
        return self.runtime


def write_corpus(
    tmp_path: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    whitelist: list[int] | None = None,
    name: str = "corpus",
) -> tuple[Path, Path]:
    """Ghi corpus và inventory tổng hợp rồi trả hai đường dẫn"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp_path / f"{name}.json"
    inventory_path = tmp_path / f"{name}_inventory.json"
    corpus_path.write_text(
        json.dumps(rows if rows is not None else default_corpus_rows(), ensure_ascii=False),
        encoding="utf-8",
    )
    inventory_path.write_text(
        json.dumps(whitelist if whitelist is not None else WHITELIST),
        encoding="utf-8",
    )
    return corpus_path, inventory_path


@dataclass(frozen=True, slots=True)
class CaptureFixture:
    """Capture bundle tổng hợp cùng input và runtime giả đã dùng"""

    outcome: StopCaptureOutcome
    corpus_path: Path
    inventory_path: Path
    runtime: FakeRuntime
    factory: FakeFactory
    settings: SinglePassSettings

    @property
    def capture_dir(self) -> Path:
        return self.outcome.capture_dir

    @property
    def drafts(self) -> list[StopPolicyCase]:
        return list(self.outcome.draft_cases)

    def case_for(self, question_id: Any, *, source_hop: int = 0) -> StopPolicyCase:
        """Lấy draft theo định danh câu hỏi và hop"""
        for case in self.drafts:
            if case.question_id == question_id and case.source_hop == source_hop:
                return case
        raise KeyError(f"no draft for question {question_id!r} at hop {source_hop}")


def capture_bundle(
    tmp_path: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    answers: dict[str, tuple[list[str], set[int]]] | None = None,
    whitelist: list[int] | None = None,
    name: str = "capture",
    settings: SinglePassSettings | None = None,
    error: Exception | None = None,
) -> CaptureFixture:
    """Chạy capture tổng hợp với runtime giả và trả bundle đã publish"""
    corpus_path, inventory_path = write_corpus(
        tmp_path, rows=rows, whitelist=whitelist, name=f"{name}-corpus"
    )
    runtime = FakeRuntime(answers if answers is not None else default_answers(), error=error)
    factory = FakeFactory(runtime)
    effective_settings = settings if settings is not None else SinglePassSettings()
    outcome = capture_corpus(
        corpus_path,
        inventory_path,
        output_dir=tmp_path / name,
        settings=effective_settings,
        runtime_factory=factory,
    )
    return CaptureFixture(
        outcome=outcome,
        corpus_path=corpus_path,
        inventory_path=inventory_path,
        runtime=runtime,
        factory=factory,
        settings=effective_settings,
    )


def approve(
    case: StopPolicyCase,
    *,
    expected_action: str,
    acceptable_dieu: list[int] | None = None,
    semantic_group_id: str,
    split: str,
    label_reason: str = "Nhãn tổng hợp dựa trên question và observation",
    fewshot_overlap: bool | None = False,
    overlap_notes: str | None = None,
) -> StopPolicyCase:
    """Duyệt một draft case tổng hợp qua validation thật của contract"""
    payload = case.model_dump(mode="json")
    payload.update(
        {
            "expected_action": expected_action,
            "acceptable_dieu": list(acceptable_dieu or []),
            "label_reason": label_reason,
            "label_status": "approved",
            "label_observation_hash": case.observation_hash,
            "semantic_group_id": semantic_group_id,
            "split": split,
            "fewshot_overlap": fewshot_overlap,
            "overlap_notes": overlap_notes,
        }
    )
    return StopPolicyCase.model_validate(payload)


def tamper(case: StopPolicyCase, **fields: Any) -> StopPolicyCase:
    """Sửa case mà không chạy lại validation, để test validator bắt lỗi tại chỗ"""
    return case.model_copy(update=fields)


STATE_FIELDS: tuple[str, ...] = (
    "dataset_id",
    "question_id",
    "source_hop",
    "question",
    "observation",
    "candidates",
    "source_run_id",
    "source_policy",
    "source_trial_id",
    "parent_case_id",
    "followed_dieu",
    "followed_article_sha256",
    "collected_article_refs",
)


def retarget(case: StopPolicyCase, **fields: Any) -> StopPolicyCase:
    """Đổi field frozen state rồi cập nhật lại observation binding, state hash và case id

    Dùng khi test cần một later-hop state *hợp lệ mới*, khác với `tamper` khi test
    cần bắt đúng lỗi binding
    """
    payload = case.model_dump(mode="json")
    payload.update(fields)
    if case.source_hop > 0:
        if "followed_dieu" in fields and "followed_article_sha256" not in fields:
            article_hash = sha256_text(f"synthetic article {payload['followed_dieu']}")
            payload["followed_article_sha256"] = article_hash
            payload["collected_article_refs"] = [
                {"dieu": payload["followed_dieu"], "sha256": article_hash}
            ]
        if "source_hop" in fields:
            refs = list(payload["collected_article_refs"])
            used = {item["dieu"] for item in refs}
            while len(refs) < payload["source_hop"]:
                previous_dieu = next(
                    value
                    for value in range(1, 1000)
                    if value not in used and value != payload["followed_dieu"]
                )
                previous_hash = sha256_text(f"synthetic article {previous_dieu}")
                refs.insert(0, {"dieu": previous_dieu, "sha256": previous_hash})
                used.add(previous_dieu)
            payload["collected_article_refs"] = refs[-payload["source_hop"] :]
        payload["observation_hash"] = sha256_text(payload["observation"])
        payload["label_observation_hash"] = payload["observation_hash"]
        state = LaterHopState.model_validate({name: payload[name] for name in STATE_FIELDS})
        payload["source_state_hash"] = state.state_hash
        payload["case_id"] = later_hop_case_id(
            payload["dataset_id"], payload["question_id"], state.state_hash
        )
    return StopPolicyCase.model_validate(payload)


def replace_case(cases: list[StopPolicyCase], replacement: StopPolicyCase) -> list[StopPolicyCase]:
    """Thay một case theo case_id, giữ nguyên thứ tự"""
    return [replacement if case.case_id == replacement.case_id else case for case in cases]


HOP0_LABELS: dict[int, tuple[str, list[int], str]] = {
    1: ("follow", [30], "g-question-one"),
    2: ("stop", [], "g-question-two"),
    3: ("follow", [33], "g-question-three"),
    4: ("stop", [], "g-question-four"),
}
HOP1_LABELS: dict[int, tuple[str, list[int], str]] = {
    1: ("follow", [32], "g-question-one"),
    2: ("stop", [], "g-question-two"),
    3: ("follow", [34], "g-question-three"),
    4: ("stop", [], "g-question-four"),
}
SPLIT_BY_QUESTION: dict[int, str] = {1: "dev", 2: "dev", 3: "heldout", 4: "heldout"}


def default_later_hop_states(dataset_id: str) -> list[dict[str, Any]]:
    """Dựng frozen later-hop state tổng hợp cho bốn câu hỏi đầu"""
    followed = {1: 30, 2: 31, 3: 33, 4: 35}
    candidates = {1: [32], 2: [35], 3: [34], 4: [33]}
    observations = {
        1: HOP1_OBSERVATION_ONE,
        2: HOP1_OBSERVATION_TWO,
        3: HOP1_OBSERVATION_THREE,
        4: HOP1_OBSERVATION_FOUR,
    }
    questions = {1: QUESTION_ONE, 2: QUESTION_TWO, 3: QUESTION_THREE, 4: QUESTION_FOUR}
    return [
        {
            "dataset_id": dataset_id,
            "question_id": question_id,
            "source_hop": 1,
            "question": questions[question_id],
            "observation": observations[question_id],
            "candidates": candidates[question_id],
            "source_run_id": "run-synthetic",
            "source_policy": "llm",
            "source_trial_id": f"{question_id}:hop0:r0",
            "parent_case_id": hop0_case_id(dataset_id, question_id),
            "followed_dieu": followed[question_id],
            "followed_article_sha256": sha256_text(
                f"synthetic article {followed[question_id]}"
            ),
            "collected_article_refs": [
                {
                    "dieu": followed[question_id],
                    "sha256": sha256_text(f"synthetic article {followed[question_id]}"),
                }
            ],
        }
        for question_id in (1, 2, 3, 4)
    ]


@dataclass(frozen=True, slots=True)
class ReviewedCorpus:
    """Bundle đã review: draft hop-0, later-hop đã import và nhãn đã duyệt"""

    fixture: CaptureFixture
    snapshot: SeedSnapshot
    corpus: StopCorpus
    seeds_path: Path
    drafts_path: Path
    imported_path: Path
    reviewed_path: Path
    cases: list[StopPolicyCase]
    approved: list[StopPolicyCase]

    @property
    def exclusions(self):
        return validate_reviewed_cases(self.snapshot, self.corpus, self.cases).exclusions

    def case_for(self, question_id: Any, *, source_hop: int = 0) -> StopPolicyCase:
        for case in self.cases:
            if case.question_id == question_id and case.source_hop == source_hop:
                return case
        raise KeyError(f"no reviewed case for question {question_id!r} at hop {source_hop}")


def reviewed_corpus(tmp_path: Path, *, name: str = "review") -> ReviewedCorpus:
    """Capture rồi review đủ bốn lớp nhãn trên cả hai split để test replay"""
    fixture = capture_bundle(tmp_path, name=name)
    seeds_path = fixture.capture_dir / "seeds.json"
    drafts_path = fixture.capture_dir / "cases_review.jsonl"
    snapshot = read_snapshot(seeds_path)
    corpus = read_stop_corpus(fixture.corpus_path)
    assert snapshot.dataset_id == fixture.corpus_path.stem

    approved_hop0: list[StopPolicyCase] = []
    for case in fixture.drafts:
        label = HOP0_LABELS.get(case.question_id)
        if label is None:
            continue
        action, acceptable, group = label
        approved_hop0.append(
            approve(
                case,
                expected_action=action,
                acceptable_dieu=acceptable,
                semantic_group_id=group,
                split=SPLIT_BY_QUESTION[case.question_id],
            )
        )
    base_cases = list(fixture.drafts)
    for case in approved_hop0:
        base_cases = replace_case(base_cases, case)

    states = [
        LaterHopState.model_validate(payload)
        for payload in default_later_hop_states(snapshot.dataset_id)
    ]
    imported_path = tmp_path / f"{name}-imported.jsonl"
    imported = import_later_hop_states(snapshot, base_cases, states, imported_path)

    approved_hop1: list[StopPolicyCase] = []
    for case in imported.cases:
        label = HOP1_LABELS.get(case.question_id)
        if label is None or case.source_hop == 0:
            continue
        action, acceptable, group = label
        approved_hop1.append(
            approve(
                case,
                expected_action=action,
                acceptable_dieu=acceptable,
                semantic_group_id=group,
                split=SPLIT_BY_QUESTION[case.question_id],
            )
        )
    reviewed = list(imported.cases)
    for case in approved_hop1:
        reviewed = replace_case(reviewed, case)
    reviewed_path = tmp_path / f"{name}-reviewed.jsonl"
    write_new_stop_cases(reviewed_path, reviewed)

    approved = [
        case for case in reviewed if case.label_status == "approved"
    ]
    return ReviewedCorpus(
        fixture=fixture,
        snapshot=snapshot,
        corpus=corpus,
        seeds_path=seeds_path,
        drafts_path=drafts_path,
        imported_path=imported_path,
        reviewed_path=reviewed_path,
        cases=reviewed,
        approved=approved,
    )


__all__ = [
    "CORPUS_ANSWER_MARKER",
    "CORPUS_GOLD_MARKER",
    "CORPUS_LINK_MARKER",
    "CaptureFixture",
    "FakeFactory",
    "FakeRuntime",
    "HOP0_LABELS",
    "HOP1_LABELS",
    "HOP1_OBSERVATION_FOUR",
    "HOP1_OBSERVATION_ONE",
    "HOP1_OBSERVATION_THREE",
    "HOP1_OBSERVATION_TWO",
    "QUESTION_FIVE",
    "QUESTION_FOUR",
    "QUESTION_ONE",
    "QUESTION_SIX",
    "QUESTION_THREE",
    "QUESTION_TWO",
    "ReviewedCorpus",
    "SPLIT_BY_QUESTION",
    "WHITELIST",
    "approve",
    "capture_bundle",
    "default_answers",
    "default_corpus_rows",
    "default_later_hop_states",
    "replace_case",
    "retarget",
    "reviewed_corpus",
    "tamper",
    "write_corpus",
]
