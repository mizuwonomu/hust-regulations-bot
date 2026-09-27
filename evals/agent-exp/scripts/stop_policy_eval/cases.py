"""Đọc, ghi và validate STOP case cùng nguồn corpus đã băm

Chỉ `id` và `user_input` của corpus là input; các field annotation khác chỉ vào
row hash làm provenance. Case được ghi theo cơ chế không ghi đè, và định danh
hop-0 dùng đúng công thức canonical của `seed_cases`
"""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts import GateInput, SeedSnapshot, SourceFile
from seed_cases import canonical_json, validate_dataset

from diagnostic_subexp.shared.contracts import sha256_bytes, sha256_text, structured_hash
from diagnostic_subexp.shared.source_refs import repository_root, sha256_file

from stop_policy_eval.contracts import (
    REQUIREMENT_CLASSES,
    SPLIT_ORDER,
    LaterHopState,
    StopExclusion,
    StopPolicyCase,
    StopReadiness,
    StopRequirementCount,
    StopReviewSelection,
    requirement_for,
    verify_case_invariants,
)
from stop_policy_eval.state import rebuild_case_state


@dataclass(frozen=True, slots=True)
class StopCorpusRow:
    """Một dòng corpus đã validate: định danh, question và hash provenance"""

    question_id: Any
    question: str
    row_hash: str


@dataclass(frozen=True, slots=True)
class StopCorpus:
    """Corpus đã băm cùng rows theo thứ tự file và bytes gốc"""

    path: Path
    sha256: str
    rows: tuple[StopCorpusRow, ...]
    raw: bytes

    def by_identity(self) -> dict[tuple[type[Any], Any], StopCorpusRow]:
        """Map row theo identity có kiểu của question id"""
        return {(type(row.question_id), row.question_id): row for row in self.rows}


def _typed_identity(value: Any) -> tuple[type[Any], Any]:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"question id must be an integer or string: {value!r}")
    return type(value), value


def snapshot_content_bytes(snapshot: SeedSnapshot) -> bytes:
    """Serialize snapshot thành bytes tất định để ghi seeds.json và băm"""
    return canonical_json(snapshot.model_dump(mode="json"))


def snapshot_content_hash(snapshot: SeedSnapshot) -> str:
    """Băm snapshot capture; hash này là snapshot_hash mà case tham chiếu"""
    return sha256_bytes(snapshot_content_bytes(snapshot))


def parse_stop_corpus(raw: bytes, *, path_label: str = "<memory>") -> StopCorpus:
    """Parse corpus bytes đã băm; chỉ id và user_input được xem là input

    Params:
    - raw: bytes gốc của file corpus, dùng cho dataset hash
    - path_label: nhãn đường dẫn cho thông báo lỗi
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path_label}: file is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path_label}: invalid JSON at line {exc.lineno}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path_label}: corpus must be a non-empty JSON array")

    questions = validate_dataset(Path(path_label), data)
    rows = tuple(
        StopCorpusRow(
            question_id=identity[1],
            question=question,
            row_hash=structured_hash(row),
        )
        for (identity, question), row in zip(questions.items(), data)
    )
    return StopCorpus(
        path=Path(path_label),
        sha256=sha256_bytes(raw),
        rows=rows,
        raw=raw,
    )


def read_stop_corpus(path: Path) -> StopCorpus:
    """Đọc và băm corpus từ đĩa, không đọc lại lần hai"""
    path = Path(path)
    return parse_stop_corpus(path.read_bytes(), path_label=str(path))


def hop0_case_id(dataset_id: str, question_id: Any) -> str:
    """Dựng định danh hop-0 canonical từ dataset và question id có kiểu"""
    token = sha256_bytes(canonical_json(question_id))[:12]
    return f"{dataset_id}:hop0:q-{token}"


def later_hop_case_id(dataset_id: str, question_id: Any, state_hash: str) -> str:
    """Dựng định danh later-hop từ question id và hash của frozen state"""
    token = sha256_bytes(canonical_json(question_id))[:12]
    return f"{dataset_id}:hop:q-{token}.{state_hash[:12]}"


def source_file(path: Path) -> SourceFile:
    """Băm một nguồn case/snapshot và lưu nó tương đối repository khi có thể"""
    resolved = Path(path).resolve()
    root = repository_root().resolve()
    try:
        stored = resolved.relative_to(root).as_posix()
    except ValueError:
        # Nguồn ngoài repository vẫn dùng được, nhưng phải ghi đường dẫn tuyệt đối
        stored = str(resolved)
    return SourceFile(path=stored, sha256=sha256_file(resolved))


def resolve_source_file(reference: SourceFile) -> Path:
    """Resolve một nguồn đã lưu rồi kiểm lại hash bytes của nó"""
    candidate = Path(reference.path)
    resolved = candidate if candidate.is_absolute() else repository_root() / candidate
    if not resolved.is_file():
        raise FileNotFoundError(f"required replay source is missing: {reference.path}")
    actual = sha256_file(resolved)
    if actual != reference.sha256:
        raise ValueError(
            f"{reference.path}: source hash mismatch, expected {reference.sha256}, got {actual}"
        )
    return resolved


def label_free_input(case: StopPolicyCase) -> GateInput:
    """Chiếu một case thành input không nhãn cho request rendering

    Request chỉ được dựng từ GateInput, nên nhãn và acceptable article không thể
    đi vào prompt
    """
    if not case.candidates:
        raise ValueError(f"{case.case_id}: a gate input requires non-empty candidates")
    return GateInput(
        question=case.question,
        observation=case.observation,
        candidates=list(case.candidates),
    )


def cases_bytes(cases: list[StopPolicyCase]) -> bytes:
    """Serialize cases thành JSONL tất định để băm và ghi theo cơ chế bất biến"""
    payload = bytearray()
    for case in cases:
        payload.extend(canonical_json(case.model_dump(mode="json")))
        payload.extend(b"\n")
    return bytes(payload)


def read_stop_cases(path: Path) -> list[StopPolicyCase]:
    """Đọc và validate dataset case STOP dạng JSONL, giữ nguyên thứ tự"""
    path = Path(path)
    cases: list[StopPolicyCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"{path}: blank line at {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON at line {line_number}") from exc
            try:
                case = StopPolicyCase.model_validate(payload)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid case at line {line_number}: {exc}") from exc
            if case.case_id in seen:
                raise ValueError(f"{path}: duplicate case_id at line {line_number}: {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: case file must not be empty")
    return cases


def write_new_stop_cases(path: Path, cases: list[StopPolicyCase]) -> Path:
    """Ghi case vào một đích mới, từ chối mọi đích đã tồn tại"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cases_bytes(cases)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        # Hard link qua os.link tạo đích atomic và thất bại nếu đích đã tồn tại
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise ValueError(f"{path}: destination already exists") from None
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_later_hop_states(
    path: Path,
    *,
    allow_empty: bool = False,
) -> list[LaterHopState]:
    """Đọc frozen later-hop state do người dùng cung cấp, dạng JSON array hoặc JSONL"""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    payloads: list[Any] = []
    if stripped.startswith("["):
        try:
            payloads = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON at line {exc.lineno}") from exc
        if not isinstance(payloads, list):
            raise ValueError(f"{path}: later-hop states must be a JSON array")
    else:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"{path}: blank line at {line_number}")
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON at line {line_number}") from exc
    if not payloads and not allow_empty:
        raise ValueError(f"{path}: later-hop state file must not be empty")

    states: list[LaterHopState] = []
    seen: set[tuple[type[Any], Any, int, str]] = set()
    for index, payload in enumerate(payloads):
        recorded_hash = None
        if isinstance(payload, dict):
            payload = dict(payload)
            recorded_hash = payload.pop("state_hash", None)
        try:
            state = LaterHopState.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid later-hop state {index}: {exc}") from exc
        if recorded_hash is not None and recorded_hash != state.state_hash:
            raise ValueError(f"{path}: later-hop state {index} hash does not match its fields")
        identity = (
            _typed_identity(state.question_id)[0],
            state.question_id,
            state.source_hop,
            state.state_hash,
        )
        if identity in seen:
            raise ValueError(f"{path}: duplicate later-hop state {index} for one question")
        seen.add(identity)
        states.append(state)
    return states


@dataclass(frozen=True, slots=True)
class StopImportOutcome:
    """Case file mới sau khi nhập frozen later-hop state"""

    output_path: Path
    cases: list[StopPolicyCase]
    added_case_ids: list[str]


def _identity(value: Any) -> tuple[type[Any], Any]:
    return type(value), value


def _case_state_key(case: StopPolicyCase) -> tuple[Any, ...]:
    """Khóa semantic state của một case để phát hiện state trùng"""
    if case.source_hop == 0:
        return (_identity(case.question_id), 0, hop0_case_id(case.dataset_id, case.question_id))
    return (_identity(case.question_id), case.source_hop, case.source_state_hash)


def _exclusion_reason(case: StopPolicyCase) -> str | None:
    """Lý do loại một case khỏi schedule, theo thứ tự ưu tiên cơ học"""
    if not case.candidates:
        return "no_candidates"
    if case.label_status != "approved":
        return "draft"
    if case.expected_action == "unresolved":
        return "unresolved"
    if case.fewshot_overlap is None:
        return "overlap_unreviewed"
    if case.fewshot_overlap:
        return "fewshot_overlap"
    return None


def _readiness_for(split: str, eligible: list[StopPolicyCase]) -> StopReadiness:
    """Đếm case đủ điều kiện theo bốn lớp yêu cầu của một split"""
    counts = {requirement: 0 for requirement in REQUIREMENT_CLASSES}
    for case in eligible:
        if case.split != split or case.expected_action not in {"follow", "stop"}:
            continue
        counts[requirement_for(case)] += 1
    rows = [
        StopRequirementCount(
            requirement=requirement,
            eligible_cases=counts[requirement],
            satisfied=counts[requirement] > 0,
        )
        for requirement in REQUIREMENT_CLASSES
    ]
    return StopReadiness(
        split=split,
        requirements=rows,
        complete=all(row.satisfied for row in rows),
    )


def _validate_sources(
    case: StopPolicyCase,
    snapshot: SeedSnapshot,
    snapshot_hash: str,
    corpus: StopCorpus | None,
    corpus_rows: dict[tuple[type[Any], Any], StopCorpusRow],
    snapshot_rows: dict[tuple[type[Any], Any], Any],
    cases_by_id: dict[str, StopPolicyCase],
) -> None:
    """Đối chiếu một case với snapshot, corpus và parent case của nó"""
    if case.dataset_id != snapshot.dataset_id:
        raise ValueError(f"{case.case_id}: dataset_id does not match the snapshot")
    if case.snapshot_id != snapshot.snapshot_id:
        raise ValueError(f"{case.case_id}: snapshot_id does not match the snapshot")
    if case.snapshot_hash != snapshot_hash:
        raise ValueError(f"{case.case_id}: snapshot hash is stale")

    identity = _identity(case.question_id)
    row = snapshot_rows.get(identity)
    if row is None:
        raise ValueError(f"{case.case_id}: question_id is absent from the snapshot")
    if case.question != row.question:
        raise ValueError(f"{case.case_id}: question does not match the snapshot row")
    if corpus is not None:
        corpus_row = corpus_rows.get(identity)
        if corpus_row is None:
            raise ValueError(f"{case.case_id}: question_id is absent from the corpus")
        if corpus_row.question != row.question:
            raise ValueError(f"{case.case_id}: question does not match the snapshot row")
        if case.source_row_hash != corpus_row.row_hash:
            raise ValueError(f"{case.case_id}: source row hash does not match the corpus")
        if case.source_dataset_hash != corpus.sha256:
            raise ValueError(f"{case.case_id}: source dataset hash does not match the corpus")

    if case.source_hop == 0:
        expected_id = hop0_case_id(snapshot.dataset_id, case.question_id)
        if case.case_id != expected_id:
            raise ValueError(
                f"{case.case_id}: case_id does not match the canonical hop-0 identity {expected_id}"
            )
        observation, candidates = rebuild_case_state(snapshot, row)
        if case.observation != observation:
            raise ValueError(f"{case.case_id}: observation is stale or mismatched")
        if case.observation_hash != sha256_text(observation):
            raise ValueError(f"{case.case_id}: observation_hash is invalid")
        if list(case.candidates) != list(candidates):
            raise ValueError(f"{case.case_id}: candidate order is stale or mismatched")
        return

    outside = sorted(set(case.candidates) - snapshot.internal_dieu)
    if outside:
        raise ValueError(f"{case.case_id}: candidates outside the corpus whitelist: {outside}")
    parent = cases_by_id.get(case.parent_case_id)
    if parent is None:
        raise ValueError(
            f"{case.case_id}: later-hop parent case {case.parent_case_id} is absent from the case file"
        )
    if parent.source_hop != case.source_hop - 1:
        raise ValueError(f"{case.case_id}: later-hop parent case must immediately precede this state")
    if _identity(parent.question_id) != identity:
        raise ValueError(f"{case.case_id}: later-hop parent case belongs to a different question")
    if case.followed_dieu not in parent.candidates:
        raise ValueError(f"{case.case_id}: followed article is not among the parent candidates")
    expected_id = later_hop_case_id(case.dataset_id, case.question_id, case.source_state_hash)
    if case.case_id != expected_id:
        raise ValueError(
            f"{case.case_id}: case_id does not match the derived later-hop identity {expected_id}"
        )


def _validate_grouping(cases: list[StopPolicyCase]) -> None:
    """Buộc group và split đã gán nhất quán kể cả khi action label còn draft"""
    group_split: dict[str, str] = {}
    question_scope: dict[tuple[type[Any], Any], tuple[str, str]] = {}
    for case in cases:
        if case.semantic_group_id is None or case.split == "unassigned":
            continue
        scope = (case.semantic_group_id, case.split)
        known_split = group_split.setdefault(case.semantic_group_id, case.split)
        if known_split != case.split:
            raise ValueError(
                f"{case.case_id}: a semantic group cannot cross dev and heldout "
                f"({case.semantic_group_id})"
            )
        identity = _identity(case.question_id)
        known_scope = question_scope.setdefault(identity, scope)
        if known_scope != scope:
            raise ValueError(
                f"{case.case_id}: cases of the same question must keep the same semantic group "
                "and split"
            )


def validate_reviewed_cases(
    snapshot: SeedSnapshot,
    corpus: StopCorpus | None,
    cases: list[StopPolicyCase],
) -> StopReviewSelection:
    """Kiểm tra case đã review rồi tách case chạy được khỏi exclusion và readiness

    Params:
    - snapshot: snapshot v2 canonical mà case tham chiếu
    - corpus: corpus đã băm để kiểm source binding của từng row; None khi reload
      offline, lúc đó chỉ hash dataset của case được kiểm
    - cases: case đọc từ file review, không bị hàm này chỉnh sửa
    """
    if corpus is not None and corpus.sha256 != snapshot.dataset_source.sha256:
        raise ValueError("corpus bytes do not match the captured snapshot dataset hash")
    snapshot_hash = snapshot_content_hash(snapshot)
    corpus_rows = {} if corpus is None else corpus.by_identity()
    snapshot_rows = {_identity(row.id): row for row in snapshot.rows}
    cases_by_id = {case.case_id: case for case in cases}

    seen_case_ids: set[str] = set()
    for case in cases:
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)

    snapshot_identities = Counter(_identity(row.id) for row in snapshot.rows)
    hop0_identities = Counter(
        _identity(case.question_id) for case in cases if case.source_hop == 0
    )
    missing = snapshot_identities.keys() - hop0_identities.keys()
    extra = hop0_identities.keys() - snapshot_identities.keys()
    duplicate = {identity for identity, count in hop0_identities.items() if count != 1}
    duplicate.update(
        identity for identity, count in snapshot_identities.items() if count != 1
    )
    if missing or extra or duplicate:
        raise ValueError(
            "hop-0 roster does not match snapshot identities: "
            f"missing={sorted(missing, key=repr)!r}, "
            f"extra={sorted(extra, key=repr)!r}, "
            f"duplicate={sorted(duplicate, key=repr)!r}"
        )

    seen_states: set[tuple[Any, ...]] = set()
    for case in cases:
        state_key = _case_state_key(case)
        if state_key in seen_states:
            label = "duplicate state" if case.source_hop == 0 else "duplicate later-hop state"
            raise ValueError(f"{case.case_id}: {label} in one case file")
        seen_states.add(state_key)

    for case in cases:
        verify_case_invariants(case)
        _validate_sources(
            case,
            snapshot,
            snapshot_hash,
            corpus,
            corpus_rows,
            snapshot_rows,
            cases_by_id,
        )
    _validate_grouping(cases)

    eligible: list[StopPolicyCase] = []
    exclusions: list[StopExclusion] = []
    for case in cases:
        reason = _exclusion_reason(case)
        if reason is None:
            eligible.append(case)
        else:
            exclusions.append(StopExclusion(case_id=case.case_id, reason=reason))

    return StopReviewSelection(
        eligible_cases=eligible,
        exclusions=exclusions,
        readiness=[_readiness_for(split, eligible) for split in SPLIT_ORDER],
    )


def import_later_hop_states(
    snapshot: SeedSnapshot,
    base_cases: list[StopPolicyCase],
    states: list[LaterHopState],
    output: Path,
) -> StopImportOutcome:
    """Nhập frozen later-hop state do người dùng cấp thành một case version mới

    Params:
    - snapshot: snapshot canonical để kiểm định danh và inventory
    - base_cases: case hiện có, không bị hàm này chỉnh sửa
    - states: frozen state kèm provenance đầy đủ, không dựng lại từ hop-0
    - output: đích mới, bị từ chối nếu đã tồn tại
    """
    output = Path(output)
    if output.exists():
        raise ValueError(f"{output}: destination already exists")

    by_id: dict[str, StopPolicyCase] = {}
    for case in base_cases:
        if case.case_id in by_id:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        by_id[case.case_id] = case
    snapshot_rows = {_identity(row.id): row for row in snapshot.rows}
    snapshot_hash = snapshot_content_hash(snapshot)
    seen_states = {
        _case_state_key(case) for case in base_cases if case.source_hop > 0
    }

    added: list[StopPolicyCase] = []
    added_ids: list[str] = []
    for state in states:
        if state.dataset_id != snapshot.dataset_id:
            raise ValueError(
                f"later-hop state for {state.question_id!r}: dataset_id does not match the snapshot"
            )
        identity = _identity(state.question_id)
        row = snapshot_rows.get(identity)
        if row is None:
            raise ValueError(
                f"later-hop state for {state.question_id!r}: question_id is absent from the snapshot"
            )
        if state.question != row.question:
            raise ValueError(
                f"later-hop state for {state.question_id!r}: question does not match the snapshot row"
            )
        outside = sorted(set(state.candidates) - snapshot.internal_dieu)
        if outside:
            raise ValueError(
                f"later-hop state for {state.question_id!r}: candidates outside the corpus "
                f"whitelist: {outside}"
            )
        parent = by_id.get(state.parent_case_id)
        if parent is None:
            raise ValueError(
                f"later-hop state for {state.question_id!r}: parent case "
                f"{state.parent_case_id} is absent from the case file"
            )
        if parent.source_hop != state.source_hop - 1:
            raise ValueError(
                f"later-hop state for {state.question_id!r}: parent case must immediately "
                "precede this state"
            )
        if _identity(parent.question_id) != identity:
            raise ValueError(
                f"later-hop state for {state.question_id!r}: parent case belongs to a different question"
            )
        if state.followed_dieu not in parent.candidates:
            raise ValueError(
                f"later-hop state for {state.question_id!r}: followed article is not among "
                "the parent candidates"
            )
        if list(state.collected_article_refs[:-1]) != list(parent.collected_article_refs):
            raise ValueError(
                f"later-hop state for {state.question_id!r}: collected article references "
                "do not extend the parent state"
            )
        case_id = later_hop_case_id(state.dataset_id, state.question_id, state.state_hash)
        if case_id in by_id:
            raise ValueError(f"duplicate case_id: {case_id}")
        state_key = (identity, state.source_hop, state.state_hash)
        if state_key in seen_states:
            raise ValueError(
                f"{case_id}: duplicate later-hop state in one case file"
            )
        seen_states.add(state_key)

        case = StopPolicyCase(
            case_id=case_id,
            dataset_id=snapshot.dataset_id,
            question_id=state.question_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_hash=snapshot_hash,
            question=state.question,
            observation=state.observation,
            observation_hash=sha256_text(state.observation),
            candidates=list(state.candidates),
            source_hop=state.source_hop,
            source_run_id=state.source_run_id,
            source_policy=state.source_policy,
            source_trial_id=state.source_trial_id,
            parent_case_id=state.parent_case_id,
            followed_dieu=state.followed_dieu,
            source_state_hash=state.state_hash,
            followed_article_sha256=state.followed_article_sha256,
            collected_article_refs=list(state.collected_article_refs),
            expected_action="unresolved",
            acceptable_dieu=[],
            label_reason=None,
            label_status="draft",
            label_observation_hash=None,
            semantic_group_id=None,
            split="unassigned",
            fewshot_overlap=None,
            overlap_notes=None,
            source_row_hash=parent.source_row_hash,
            source_dataset_hash=parent.source_dataset_hash,
        )
        verify_case_invariants(case)
        by_id[case_id] = case
        added.append(case)
        added_ids.append(case_id)

    cases = [*base_cases, *added]
    write_new_stop_cases(output, cases)
    return StopImportOutcome(output_path=output, cases=cases, added_case_ids=added_ids)


__all__ = [
    "StopCorpus",
    "StopCorpusRow",
    "StopImportOutcome",
    "cases_bytes",
    "hop0_case_id",
    "import_later_hop_states",
    "label_free_input",
    "later_hop_case_id",
    "parse_stop_corpus",
    "read_later_hop_states",
    "read_stop_cases",
    "read_stop_corpus",
    "resolve_source_file",
    "snapshot_content_bytes",
    "snapshot_content_hash",
    "source_file",
    "validate_reviewed_cases",
    "write_new_stop_cases",
]
