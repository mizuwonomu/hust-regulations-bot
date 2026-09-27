"""Contract STOP-local cho capture, label review, replay và metric của stop policy

Package này giữ boundary giữa state gate không nhãn (question, observation,
candidates), nhãn do người duyệt và output của model. Các record dùng chung chỉ
được import lại khi ý nghĩa không đổi: GateInput, DecisionOutcome, TrialError,
Usage, RatioMetric, MeanMetric, SourceFile, message, execution config và result
compact của diagnostic
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from contracts import (
    ExpectedAction,
    LabelStatus,
    MeanMetric,
    RatioMetric,
    SourceFile,
)
from diagnostic_subexp.shared.contracts import (
    ArticleId,
    ArticleList,
    DiagnosticExecutionConfig,
    DiagnosticModel,
    DiagnosticResult,
    ManifestStatus,
    MessageRecord,
    MetadataMap,
    NonBlank,
    NonEmptyArticleList,
    NonNegativeInt,
    PolicyName,
    PositiveInt,
    TypedQuestionId,
    sha256_text,
    structured_hash,
)

STOP_POLICY_SCHEMA_VERSION = 1
STOP_EXPERIMENT_NAME = "stop-policy"
BASELINE_VARIANT = "baseline"
# Chỉ baseline được thực thi trong milestone này; các variant còn lại chưa đăng ký
VARIANT_RECIPES: dict[str, str] = {
    BASELINE_VARIANT: "current production prompt, gate renderers and thinking settings",
}


def resolve_variant(name: str) -> str:
    """Kiểm tra một prompt variant đã được đăng ký để chạy, từ chối tên lạ"""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("variant must be a non-empty string")
    if name not in VARIANT_RECIPES:
        raise ValueError(
            f"unsupported variant: {name!r}; only {sorted(VARIANT_RECIPES)} is executable "
            "in this milestone"
        )
    return name

StopSplit: TypeAlias = Literal["unassigned", "dev", "heldout"]
AssignedSplit: TypeAlias = Literal["dev", "heldout"]
StopLabel: TypeAlias = Literal["follow", "stop"]
RunPurpose: TypeAlias = Literal["evaluation", "later_hop_capture"]
StopHopClass: TypeAlias = Literal["hop0", "post_follow"]
StopRequirement: TypeAlias = Literal[
    "hop0_follow",
    "hop0_stop",
    "post_follow_follow",
    "post_follow_stop",
]
StopExclusionReason: TypeAlias = Literal[
    "draft",
    "unresolved",
    "no_candidates",
    "overlap_unreviewed",
    "fewshot_overlap",
    "split_not_selected",
    "hop_not_selected",
]
StopStrataKind: TypeAlias = Literal["case", "semantic_group", "source_hop", "split"]
CaptureStatus: TypeAlias = Literal["captured"]

REQUIREMENT_CLASSES: tuple[StopRequirement, ...] = (
    "hop0_follow",
    "hop0_stop",
    "post_follow_follow",
    "post_follow_stop",
)
# Requirement nào đòi hop-0 và requirement nào đòi state sau một follow
POST_FOLLOW_REQUIREMENTS: tuple[StopRequirement, ...] = (
    "post_follow_follow",
    "post_follow_stop",
)
SPLIT_ORDER: tuple[AssignedSplit, ...] = ("dev", "heldout")


def hop_class_for(source_hop: int) -> StopHopClass:
    """Phân lớp state theo hop: hop 0 trước follow đầu tiên, còn lại sau follow."""
    return "hop0" if source_hop == 0 else "post_follow"


def requirement_for(case: StopPolicyCase) -> StopRequirement:
    """Xếp một case đã duyệt vào đúng lớp yêu cầu của ma trận readiness."""
    if case.expected_action not in {"follow", "stop"}:
        raise ValueError(f"{case.case_id}: readiness requires a follow or stop label")
    prefix = "hop0" if case.source_hop == 0 else "post_follow"
    return f"{prefix}_{case.expected_action}"  # type: ignore[return-value]


def frozen_state_payload(
    *,
    dataset_id: str,
    question_id: Any,
    source_hop: int,
    question: str,
    observation: str,
    candidates: list[int],
    source_run_id: str,
    source_policy: str,
    source_trial_id: str,
    parent_case_id: str,
    followed_dieu: int,
    followed_article_sha256: str,
    collected_article_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Dựng payload frozen state để băm, chỉ gồm định danh và state đã đóng băng"""
    return {
        "dataset_id": dataset_id,
        "question_id": question_id,
        "source_hop": source_hop,
        "question": question,
        "observation": observation,
        "candidates": list(candidates),
        "source_run_id": source_run_id,
        "source_policy": source_policy,
        "source_trial_id": source_trial_id,
        "parent_case_id": parent_case_id,
        "followed_dieu": followed_dieu,
        "followed_article_sha256": followed_article_sha256,
        "collected_article_refs": list(collected_article_refs),
    }


def later_hop_state_hash(case: StopPolicyCase) -> str:
    """Băm frozen state của một later-hop case từ chính field đã lưu"""
    if case.source_hop <= 0:
        raise ValueError(f"{case.case_id}: only later-hop cases carry a frozen state hash")
    missing = [
        name
        for name in (
            "source_run_id",
            "source_policy",
            "source_trial_id",
            "parent_case_id",
            "followed_dieu",
            "followed_article_sha256",
            "source_state_hash",
        )
        if getattr(case, name) is None
    ]
    if missing:
        raise ValueError(f"{case.case_id}: later-hop cases require {', '.join(missing)}")
    return structured_hash(
        frozen_state_payload(
            dataset_id=case.dataset_id,
            question_id=case.question_id,
            source_hop=case.source_hop,
            question=case.question,
            observation=case.observation,
            candidates=list(case.candidates),
            source_run_id=case.source_run_id,  # type: ignore[arg-type]
            source_policy=case.source_policy,  # type: ignore[arg-type]
            source_trial_id=case.source_trial_id,  # type: ignore[arg-type]
            parent_case_id=case.parent_case_id,  # type: ignore[arg-type]
            followed_dieu=case.followed_dieu,  # type: ignore[arg-type]
            followed_article_sha256=case.followed_article_sha256,  # type: ignore[arg-type]
            collected_article_refs=[
                item.model_dump(mode="json") for item in case.collected_article_refs
            ],
        )
    )


class CollectedArticleReference(DiagnosticModel):
    """Điều đã follow cần được fetch lại để dựng successor state"""

    dieu: ArticleId
    sha256: NonBlank

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("article reference sha256 must be a lowercase SHA-256 digest")
        return value


class LaterHopState(DiagnosticModel):
    """Frozen state sau một follow, do người dùng cung cấp kèm provenance đầy đủ"""

    dataset_id: NonBlank
    question_id: TypedQuestionId
    source_hop: NonNegativeInt
    question: NonBlank
    observation: str
    candidates: NonEmptyArticleList
    source_run_id: NonBlank
    source_policy: NonBlank
    source_trial_id: NonBlank
    parent_case_id: NonBlank
    followed_dieu: ArticleId
    followed_article_sha256: NonBlank
    collected_article_refs: list[CollectedArticleReference]

    @field_validator("observation")
    @classmethod
    def _observation(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("observation must be a string")
        return value

    @model_validator(mode="after")
    def _state_invariant(self) -> LaterHopState:
        if self.source_hop <= 0:
            raise ValueError("a later-hop state requires a positive hop")
        if len(self.collected_article_refs) != self.source_hop:
            raise ValueError("collected article references must cover every followed hop")
        if len({item.dieu for item in self.collected_article_refs}) != len(
            self.collected_article_refs
        ):
            raise ValueError("collected article references must have unique article ids")
        if not self.collected_article_refs:
            raise ValueError("a later-hop state requires its followed article reference")
        last_article = self.collected_article_refs[-1]
        if (
            last_article.dieu != self.followed_dieu
            or last_article.sha256 != self.followed_article_sha256
        ):
            raise ValueError("last collected article reference must match the followed article")
        return self

    def payload(self) -> dict[str, Any]:
        """Payload frozen state để băm, không gồm nhãn hay output của model"""
        return frozen_state_payload(
            dataset_id=self.dataset_id,
            question_id=self.question_id,
            source_hop=self.source_hop,
            question=self.question,
            observation=self.observation,
            candidates=list(self.candidates),
            source_run_id=self.source_run_id,
            source_policy=self.source_policy,
            source_trial_id=self.source_trial_id,
            parent_case_id=self.parent_case_id,
            followed_dieu=self.followed_dieu,
            followed_article_sha256=self.followed_article_sha256,
            collected_article_refs=[
                item.model_dump(mode="json") for item in self.collected_article_refs
            ],
        )

    @property
    def state_hash(self) -> str:
        """Băm toàn bộ frozen state để case import bind đúng observation"""
        return structured_hash(self.payload())


class LaterHopTerminalOutcome(DiagnosticModel):
    """State sau FOLLOW không có candidates nên không phát sinh gate call"""

    dataset_id: NonBlank
    question_id: TypedQuestionId
    source_hop: NonNegativeInt
    question: NonBlank
    observation: str
    observation_hash: NonBlank
    candidates: ArticleList
    source_run_id: NonBlank
    source_policy: NonBlank
    source_trial_id: NonBlank
    parent_case_id: NonBlank
    followed_dieu: ArticleId
    followed_article_sha256: NonBlank
    collected_article_refs: list[CollectedArticleReference]
    terminal_reason: Literal["empty_frontier"]
    terminal_state_hash: NonBlank

    @model_validator(mode="after")
    def _terminal_invariant(self) -> LaterHopTerminalOutcome:
        if self.source_hop <= 0:
            raise ValueError("a terminal later-hop outcome requires a positive hop")
        if self.candidates:
            raise ValueError("terminal later-hop outcomes must have an empty frontier")
        if self.observation_hash != sha256_text(self.observation):
            raise ValueError("terminal observation hash does not match its bytes")
        if len(self.collected_article_refs) != self.source_hop:
            raise ValueError("terminal collected article references must cover every followed hop")
        if not self.collected_article_refs:
            raise ValueError("a terminal outcome requires its followed article reference")
        last_article = self.collected_article_refs[-1]
        if (
            last_article.dieu != self.followed_dieu
            or last_article.sha256 != self.followed_article_sha256
        ):
            raise ValueError("last terminal article reference must match the followed article")
        payload = frozen_state_payload(
            dataset_id=self.dataset_id,
            question_id=self.question_id,
            source_hop=self.source_hop,
            question=self.question,
            observation=self.observation,
            candidates=[],
            source_run_id=self.source_run_id,
            source_policy=self.source_policy,
            source_trial_id=self.source_trial_id,
            parent_case_id=self.parent_case_id,
            followed_dieu=self.followed_dieu,
            followed_article_sha256=self.followed_article_sha256,
            collected_article_refs=[
                item.model_dump(mode="json") for item in self.collected_article_refs
            ],
        )
        if structured_hash(payload) != self.terminal_state_hash:
            raise ValueError("terminal state hash does not match its frozen fields")
        return self


def verify_case_invariants(case: StopPolicyCase) -> None:
    """Kiểm tra mọi invariant liên field của một STOP case

    Hàm tách khỏi model để validator có thể kiểm lại case dựng bằng model_copy,
    nơi pydantic không chạy lại validation
    """
    if case.observation_hash != sha256_text(case.observation):
        raise ValueError(f"{case.case_id}: observation_hash does not match the observation bytes")
    candidates = set(case.candidates)
    acceptable = set(case.acceptable_dieu)
    if not acceptable <= candidates:
        raise ValueError(f"{case.case_id}: acceptable_dieu must be a subset of candidates")
    if case.expected_action == "follow":
        if not case.candidates:
            raise ValueError(f"{case.case_id}: follow labels require candidates at this state")
        if not acceptable:
            raise ValueError(f"{case.case_id}: follow labels require a non-empty acceptable_dieu")
    elif acceptable:
        raise ValueError(f"{case.case_id}: {case.expected_action} labels require empty acceptable_dieu")
    if case.expected_action == "no_candidates" and case.candidates:
        raise ValueError(f"{case.case_id}: no_candidates cases must have an empty candidate list")

    if case.label_status == "approved":
        if case.split not in SPLIT_ORDER:
            raise ValueError(f"{case.case_id}: approved cases require dev or heldout split")
        if case.semantic_group_id is None or not case.semantic_group_id.strip():
            raise ValueError(f"{case.case_id}: approved cases require a semantic_group_id")
        if case.label_reason is None or not case.label_reason.strip():
            raise ValueError(f"{case.case_id}: approved cases require a non-blank label_reason")
        if case.label_observation_hash != case.observation_hash:
            raise ValueError(
                f"{case.case_id}: label_observation_hash must equal the approved observation hash"
            )
    else:
        if case.label_observation_hash is not None:
            raise ValueError(f"{case.case_id}: draft cases cannot carry a label observation hash")

    if case.expected_action == "unresolved" and case.label_status == "approved":
        pass  # Unresolved đã duyệt vẫn bị loại khỏi replay, contract chỉ đòi review field đầy đủ

    if case.source_hop == 0:
        later_hop_fields = {
            "source_run_id": case.source_run_id,
            "source_policy": case.source_policy,
            "source_trial_id": case.source_trial_id,
            "parent_case_id": case.parent_case_id,
            "followed_dieu": case.followed_dieu,
            "source_state_hash": case.source_state_hash,
            "followed_article_sha256": case.followed_article_sha256,
        }
        present = sorted(name for name, value in later_hop_fields.items() if value is not None)
        if case.collected_article_refs:
            present.append("collected_article_refs")
        if present:
            raise ValueError(
                f"{case.case_id}: hop-0 cases cannot carry later-hop provenance: {present}"
            )
        return

    missing = [
        name
        for name in (
            "source_run_id",
            "source_policy",
            "source_trial_id",
            "parent_case_id",
            "followed_dieu",
            "followed_article_sha256",
            "source_state_hash",
        )
        if getattr(case, name) is None
    ]
    if missing:
        raise ValueError(f"{case.case_id}: later-hop cases require {', '.join(missing)}")
    if not case.candidates:
        raise ValueError(f"{case.case_id}: later-hop cases require non-empty candidates")
    if len(case.collected_article_refs) != case.source_hop:
        raise ValueError(f"{case.case_id}: later-hop article references do not cover every hop")
    if not case.collected_article_refs:
        raise ValueError(f"{case.case_id}: later-hop cases require collected article references")
    last_article = case.collected_article_refs[-1]
    if (
        last_article.dieu != case.followed_dieu
        or last_article.sha256 != case.followed_article_sha256
    ):
        raise ValueError(f"{case.case_id}: followed article hash does not match collected references")
    if case.source_state_hash != later_hop_state_hash(case):
        raise ValueError(f"{case.case_id}: source_state_hash does not match the frozen state")


class StopPolicyCase(DiagnosticModel):
    """Một state stop-policy cùng provenance và field review của người"""

    case_id: NonBlank
    dataset_id: NonBlank
    question_id: TypedQuestionId
    snapshot_id: NonBlank
    snapshot_hash: NonBlank
    question: NonBlank
    observation: str
    observation_hash: NonBlank
    candidates: ArticleList
    source_hop: NonNegativeInt
    source_run_id: NonBlank | None = None
    source_policy: NonBlank | None = None
    source_trial_id: NonBlank | None = None
    parent_case_id: NonBlank | None = None
    followed_dieu: ArticleId | None = None
    source_state_hash: NonBlank | None = None
    followed_article_sha256: NonBlank | None = None
    collected_article_refs: list[CollectedArticleReference] = Field(default_factory=list)
    expected_action: ExpectedAction
    acceptable_dieu: ArticleList = Field(default_factory=list)
    label_reason: str | None = None
    label_status: LabelStatus
    label_observation_hash: NonBlank | None = None
    semantic_group_id: NonBlank | None = None
    split: StopSplit
    fewshot_overlap: bool | None = None
    overlap_notes: str | None = None
    source_row_hash: NonBlank
    source_dataset_hash: NonBlank

    @field_validator("observation")
    @classmethod
    def _observation(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("observation must be a string")
        return value

    @field_validator("label_reason", "overlap_notes")
    @classmethod
    def _optional_text(cls, value: str | None, info) -> str | None:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a string or null")
        return value

    @model_validator(mode="after")
    def _case_invariant(self) -> StopPolicyCase:
        verify_case_invariants(self)
        return self


class StopExclusion(DiagnosticModel):
    """Case bị loại khỏi schedule stop-policy cùng lý do xác định"""

    case_id: NonBlank
    reason: StopExclusionReason


class StopRequirementCount(DiagnosticModel):
    """Số case đủ điều kiện của một lớp yêu cầu trong một split"""

    requirement: StopRequirement
    eligible_cases: NonNegativeInt
    satisfied: bool

    @model_validator(mode="after")
    def _satisfied_invariant(self) -> StopRequirementCount:
        if self.satisfied != (self.eligible_cases > 0):
            raise ValueError("satisfied must match whether the class has eligible cases")
        return self


class StopReadiness(DiagnosticModel):
    """Ma trận readiness của một split, không nhúng quota số lượng corpus"""

    split: StopSplit
    requirements: list[StopRequirementCount]
    complete: bool

    @model_validator(mode="after")
    def _readiness_invariant(self) -> StopReadiness:
        if tuple(item.requirement for item in self.requirements) != REQUIREMENT_CLASSES:
            raise ValueError("a readiness matrix must declare exactly the four requirement classes")
        if self.complete != all(item.satisfied for item in self.requirements):
            raise ValueError("complete must match the satisfied requirement classes")
        if self.split == "unassigned":
            raise ValueError("a readiness matrix requires an assigned split")
        return self

    def missing(self) -> list[StopRequirement]:
        """Liệt kê các lớp yêu cầu chưa có case đủ điều kiện"""
        return [item.requirement for item in self.requirements if not item.satisfied]

    def count_for(self, requirement: StopRequirement) -> int:
        """Trả số case đủ điều kiện của một lớp yêu cầu"""
        for item in self.requirements:
            if item.requirement == requirement:
                return item.eligible_cases
        raise ValueError(f"unknown requirement class: {requirement}")


class StopReviewSelection(DiagnosticModel):
    """Case đã duyệt chạy được cùng exclusion và readiness theo từng split"""

    eligible_cases: list[StopPolicyCase]
    exclusions: list[StopExclusion]
    readiness: list[StopReadiness]

    @model_validator(mode="after")
    def _selection_invariant(self) -> StopReviewSelection:
        eligible_ids = [case.case_id for case in self.eligible_cases]
        if len(set(eligible_ids)) != len(eligible_ids):
            raise ValueError("eligible cases must have unique case_id values")
        if tuple(item.split for item in self.readiness) != SPLIT_ORDER:
            raise ValueError("readiness matrices must be reported for dev then heldout")
        excluded_ids = [item.case_id for item in self.exclusions]
        if len(set(excluded_ids)) != len(excluded_ids):
            raise ValueError("exclusions must have unique case_id values")
        overlap = set(eligible_ids) & set(excluded_ids)
        if overlap:
            raise ValueError(f"a case cannot be eligible and excluded at once: {sorted(overlap)}")
        return self


class StopInputTrace(DiagnosticModel):
    """Request baseline bất biến của một input replay, không chứa nhãn hay kết quả"""

    input_trace_id: NonBlank
    case_id: NonBlank
    dataset_id: NonBlank
    question_id: TypedQuestionId
    source_hop: NonNegativeInt
    hop_class: StopHopClass
    variant: NonBlank
    question: NonBlank
    observation: str
    question_hash: NonBlank
    observation_hash: NonBlank
    candidates: NonEmptyArticleList
    grammar_candidates: NonEmptyArticleList
    effective_messages: list[MessageRecord]
    messages_hash: NonBlank
    grammar_text: str
    grammar_hash: NonBlank
    input_fingerprint: NonBlank

    @field_validator("grammar_text")
    @classmethod
    def _grammar_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("grammar_text must be a non-empty string")
        return value

    @field_validator("observation")
    @classmethod
    def _observation(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("observation must be a string")
        return value

    @field_validator("effective_messages")
    @classmethod
    def _messages(cls, value: list[MessageRecord]) -> list[MessageRecord]:
        if not value:
            raise ValueError("effective_messages must not be empty")
        return value

    @model_validator(mode="after")
    def _trace_invariant(self) -> StopInputTrace:
        if self.hop_class != hop_class_for(self.source_hop):
            raise ValueError("hop_class does not match the trace source hop")
        if self.grammar_hash != sha256_text(self.grammar_text):
            raise ValueError("grammar_hash does not match the stored grammar text")
        if self.question_hash != sha256_text(self.question):
            raise ValueError("question_hash does not match the stored question")
        if self.observation_hash != sha256_text(self.observation):
            raise ValueError("observation_hash does not match the stored observation")
        expected_messages = structured_hash(
            [message.model_dump(mode="json") for message in self.effective_messages]
        )
        if self.messages_hash != expected_messages:
            raise ValueError("messages_hash does not match the stored messages")
        if list(self.candidates) != list(self.grammar_candidates):
            raise ValueError("baseline traces must render the grammar from the same candidates")
        return self


class StopTrial(DiagnosticModel):
    """Một slot replay độc lập policy, nhãn và kết quả"""

    trial_id: NonBlank
    input_trace_id: NonBlank
    case_id: NonBlank
    variant: NonBlank
    repeat_id: NonNegativeInt
    input_fingerprint: NonBlank


class StopResult(DiagnosticResult):
    """Result compact của một STOP policy trial theo contract diagnostic chung"""


class StopPlan(DiagnosticModel):
    """Lịch replay tất định đã validate cùng trace của từng input duy nhất"""

    split: AssignedSplit
    run_purpose: RunPurpose = "evaluation"
    variant: NonBlank
    policies: list[PolicyName]
    repeats: PositiveInt
    cases_source: SourceFile
    snapshot_source: SourceFile
    corpus_sha256: NonBlank
    selected_case_ids: list[NonBlank]
    traces: list[StopInputTrace]
    trials: list[StopTrial]
    exclusions: list[StopExclusion]
    readiness: StopReadiness
    expected_trial_count: NonNegativeInt
    expected_policy_result_count: NonNegativeInt

    @model_validator(mode="after")
    def _plan_invariant(self) -> StopPlan:
        if not self.policies or len(set(self.policies)) != len(self.policies):
            raise ValueError("policies must be nonempty and unique")
        if self.run_purpose == "later_hop_capture":
            if len(self.policies) != 1 or self.repeats != 1:
                raise ValueError("later-hop capture plans require one policy and one repeat")
            if any(trace.source_hop != 0 or not trace.candidates for trace in self.traces):
                raise ValueError("later-hop capture plans accept only nonempty hop-0 states")
        if len(set(self.selected_case_ids)) != len(self.selected_case_ids):
            raise ValueError("selected_case_ids must be unique")
        trace_ids = [trace.input_trace_id for trace in self.traces]
        if len(set(trace_ids)) != len(trace_ids):
            raise ValueError("plan traces must have unique input_trace_id values")
        if sorted(trace.case_id for trace in self.traces) != sorted(self.selected_case_ids):
            raise ValueError("plan traces must cover exactly the selected cases")
        if any(trace.variant != self.variant for trace in self.traces):
            raise ValueError("plan traces must carry the planned variant")
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("plan trials must have unique trial_id values")
        if self.expected_trial_count != len(self.trials):
            raise ValueError("expected_trial_count must match the planned schedule")
        if self.expected_policy_result_count != len(self.trials) * len(self.policies):
            raise ValueError("expected_policy_result_count must match the planned schedule")
        known_traces = {trace.input_trace_id: trace for trace in self.traces}
        for trial in self.trials:
            trace = known_traces.get(trial.input_trace_id)
            if trace is None:
                raise ValueError(f"trial {trial.trial_id} references an unknown input trace")
            if trial.case_id != trace.case_id:
                raise ValueError(f"trial {trial.trial_id} references a different case")
            if trial.input_fingerprint != trace.input_fingerprint:
                raise ValueError(f"trial {trial.trial_id} changes the input fingerprint")
            if trial.variant != self.variant:
                raise ValueError(f"trial {trial.trial_id} changes the planned variant")
        scheduled_case_ids = {trial.case_id for trial in self.trials}
        if scheduled_case_ids - set(self.selected_case_ids):
            raise ValueError("plan trials must reference selected cases only")
        if scheduled_case_ids & {item.case_id for item in self.exclusions}:
            raise ValueError("a case cannot be scheduled and excluded at once")
        if self.readiness.split != self.split:
            raise ValueError("plan readiness must describe the planned split")
        return self


class StopCaptureManifest(DiagnosticModel):
    """Manifest của một capture bundle chưa bị chỉnh sửa bởi người duyệt"""

    schema_version: int
    capture_status: CaptureStatus
    capture_id: NonBlank
    captured_at: NonBlank
    dataset_source: SourceFile
    inventory_source: SourceFile
    snapshot_id: NonBlank
    snapshot_hash: NonBlank
    row_count: PositiveInt
    generated_files: dict[str, NonBlank]
    retrieval_config: MetadataMap
    captured_source_hashes: dict[str, NonBlank]
    runtime_metadata: MetadataMap
    source_revision: str | None = None
    working_tree_modified: bool | None = None
    python_version: NonBlank
    unavailable_metadata: dict[str, str]

    @model_validator(mode="after")
    def _capture_invariant(self) -> StopCaptureManifest:
        if self.schema_version != STOP_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported capture schema version: {self.schema_version}, "
                f"supported version is {STOP_POLICY_SCHEMA_VERSION}"
            )
        required = {"seeds.json", "cases_draft.jsonl", "cases_review.jsonl", "label_review.md"}
        missing = sorted(required - set(self.generated_files))
        if missing:
            raise ValueError(f"capture manifests must hash every generated file: {missing}")
        if not self.captured_source_hashes:
            raise ValueError("capture manifests require the captured source hashes")
        return self


class StopRunManifest(DiagnosticModel):
    """Manifest của một prepared/replayed STOP run, chỉ tham chiếu nguồn canonical"""

    schema_version: int
    experiment: Literal["stop-policy"]
    run_purpose: RunPurpose = "evaluation"
    run_id: NonBlank
    started_at: NonBlank
    ended_at: str | None = None
    status: ManifestStatus
    variant: NonBlank
    split: AssignedSplit
    policies: list[PolicyName]
    repeats: PositiveInt
    cases_source: SourceFile
    snapshot_source: SourceFile
    corpus_sha256: NonBlank
    exclusions: list[StopExclusion]
    trials: list[StopTrial]
    input_trace_ids: list[NonBlank]
    readiness: StopReadiness
    execution_config: DiagnosticExecutionConfig | None = None
    execution_config_reason: str | None = None
    policy_config: dict[PolicyName, MetadataMap]
    expected_trial_count: NonNegativeInt
    expected_policy_result_count: NonNegativeInt
    execution_revision: NonBlank
    execution_dirty: bool
    executed_module_hashes: dict[str, NonBlank]

    @model_validator(mode="after")
    def _manifest_invariant(self) -> StopRunManifest:
        if self.schema_version != STOP_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported run schema version: {self.schema_version}, "
                f"supported version is {STOP_POLICY_SCHEMA_VERSION}"
            )
        if not self.policies or len(set(self.policies)) != len(self.policies):
            raise ValueError("policies must be nonempty and unique")
        if self.run_purpose == "later_hop_capture" and (
            len(self.policies) != 1 or self.repeats != 1
        ):
            raise ValueError("later-hop capture runs require one policy and one repeat")
        if set(self.policy_config) != set(self.policies):
            raise ValueError("policy_config keys must match policies")
        for source in (self.cases_source, self.snapshot_source):
            path = Path(source.path)
            if path.is_absolute():
                continue
            if ".." in path.parts:
                raise ValueError("relative manifest sources must not escape the repository")
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("manifest trials must have unique trial_id values")
        if len(set(self.input_trace_ids)) != len(self.input_trace_ids):
            raise ValueError("input_trace_ids must be unique")
        known_traces = set(self.input_trace_ids)
        for trial in self.trials:
            if trial.input_trace_id not in known_traces:
                raise ValueError(f"trial {trial.trial_id} references an unknown input trace")
            if trial.variant != self.variant:
                raise ValueError(f"trial {trial.trial_id} changes the manifest variant")
        if self.expected_trial_count != len(self.trials):
            raise ValueError("expected_trial_count must match the saved schedule")
        if self.expected_policy_result_count != len(self.trials) * len(self.policies):
            raise ValueError("expected_policy_result_count must match policies and trials")
        scheduled_case_ids = {trial.case_id for trial in self.trials}
        if scheduled_case_ids & {item.case_id for item in self.exclusions}:
            raise ValueError("a case cannot be scheduled and excluded at once")
        if (self.execution_config is None) == (self.execution_config_reason is None):
            raise ValueError(
                "an unrecorded execution config requires a reason; a recorded one forbids it"
            )
        if self.readiness.split != self.split:
            raise ValueError("manifest readiness must describe the manifest split")
        if not self.executed_module_hashes:
            raise ValueError("run manifests require executed module hashes")
        for module_name, digest in self.executed_module_hashes.items():
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"executed module hash for {module_name} is not a SHA-256 digest")
        return self


class StopLabelRates(DiagnosticModel):
    """Action rate hợp lệ theo từng lớp nhãn, giữ hai complement ở mức model"""

    follow_accuracy: RatioMetric
    false_stop_rate: RatioMetric
    stop_accuracy: RatioMetric
    over_hop_rate: RatioMetric
    follow_valid: NonNegativeInt
    stop_valid: NonNegativeInt

    @model_validator(mode="after")
    def _rate_invariant(self) -> StopLabelRates:
        if self.follow_accuracy.denominator != self.follow_valid:
            raise ValueError("follow_valid must match the follow-label denominator")
        if self.false_stop_rate.denominator != self.follow_valid:
            raise ValueError("follow_valid must match the follow-label denominator")
        if self.stop_accuracy.denominator != self.stop_valid:
            raise ValueError("stop_valid must match the stop-label denominator")
        if self.over_hop_rate.denominator != self.stop_valid:
            raise ValueError("stop_valid must match the stop-label denominator")
        follow_actions = self.follow_accuracy.numerator + self.false_stop_rate.numerator
        if follow_actions != self.follow_valid:
            raise ValueError("follow-label action counts must be complementary counts")
        stop_actions = self.stop_accuracy.numerator + self.over_hop_rate.numerator
        if stop_actions != self.stop_valid:
            raise ValueError("stop-label action counts must be complementary counts")
        for left, right in (
            (self.follow_accuracy, self.false_stop_rate),
            (self.stop_accuracy, self.over_hop_rate),
        ):
            if left.denominator:
                if left.value is None or right.value is None:
                    raise ValueError("non-zero denominators require finite ratios")
                if not math.isclose(left.value + right.value, 1.0, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("complementary action rates must sum to one")
        return self


class StopCaseMacro(DiagnosticModel):
    """Mean theo case của từng metric, kèm số case defined/eligible/excluded"""

    case_count: NonNegativeInt
    follow_accuracy: MeanMetric
    stop_accuracy: MeanMetric
    action_accuracy: MeanMetric
    selection_accuracy: MeanMetric
    decision_accuracy: MeanMetric


class StopPolicySummary(DiagnosticModel):
    """Coverage, action, selection và decision metric của một policy STOP"""

    scheduled: NonNegativeInt
    valid: NonNegativeInt
    errors: NonNegativeInt
    missing: NonNegativeInt
    action_accuracy: RatioMetric
    selection_accuracy: RatioMetric
    decision_accuracy: RatioMetric
    label_rates: StopLabelRates
    error_rate: dict[StopLabel, RatioMetric]
    missing_rate: dict[StopLabel, RatioMetric]
    case_macro: StopCaseMacro
    telemetry: MetadataMap

    @model_validator(mode="after")
    def _coverage_invariant(self) -> StopPolicySummary:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        if set(self.error_rate) != {"follow", "stop"}:
            raise ValueError("error rates must be reported for both labels")
        if set(self.missing_rate) != {"follow", "stop"}:
            raise ValueError("missing rates must be reported for both labels")
        if self.action_accuracy.denominator != self.scheduled:
            raise ValueError("scheduled action accuracy must cover every scheduled slot")
        if self.decision_accuracy.denominator != self.scheduled:
            raise ValueError("scheduled decision accuracy must cover every scheduled slot")
        return self


class StopStrataRow(DiagnosticModel):
    """Coverage và accuracy của một policy trong một strata"""

    kind: StopStrataKind
    value: NonBlank
    policy: PolicyName
    scheduled: NonNegativeInt
    valid: NonNegativeInt
    errors: NonNegativeInt
    missing: NonNegativeInt
    action_accuracy: RatioMetric
    selection_accuracy: RatioMetric
    decision_accuracy: RatioMetric

    @model_validator(mode="after")
    def _coverage_invariant(self) -> StopStrataRow:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        return self


class StopPairedSummary(DiagnosticModel):
    """So sánh hai policy trên cùng trial ID, tách action agreement khỏi decision"""

    policies: list[PolicyName]
    scheduled_pairs: NonNegativeInt
    valid_pairs: NonNegativeInt
    incomplete_pairs: NonNegativeInt
    action_agreement: RatioMetric
    a_wins: NonNegativeInt
    b_wins: NonNegativeInt
    ties: NonNegativeInt
    different_trial_ids: list[NonBlank]

    @model_validator(mode="after")
    def _paired_invariant(self) -> StopPairedSummary:
        if len(self.policies) != 2 or len(set(self.policies)) != 2:
            raise ValueError("paired summaries require two distinct policies")
        if self.valid_pairs + self.incomplete_pairs != self.scheduled_pairs:
            raise ValueError("valid_pairs and incomplete_pairs must partition scheduled_pairs")
        if self.a_wins + self.b_wins + self.ties != self.valid_pairs:
            raise ValueError("paired correctness counts must partition valid_pairs")
        if self.action_agreement.denominator != self.valid_pairs:
            raise ValueError("action agreement must be measured over valid pairs")
        if len(set(self.different_trial_ids)) != len(self.different_trial_ids):
            raise ValueError("different_trial_ids must be unique")
        return self


class StopSummary(DiagnosticModel):
    """Aggregate tất định của một STOP run, tính lại được khi offline"""

    schema_version: int
    experiment: Literal["stop-policy"]
    run_id: NonBlank
    variant: NonBlank
    split: AssignedSplit
    policies: list[PolicyName]
    scheduled: NonNegativeInt
    valid: NonNegativeInt
    errors: NonNegativeInt
    missing: NonNegativeInt
    by_policy: dict[PolicyName, StopPolicySummary]
    strata: list[StopStrataRow]
    readiness: StopReadiness
    paired: StopPairedSummary | None = None
    exclusions: list[StopExclusion]

    @model_validator(mode="after")
    def _summary_invariant(self) -> StopSummary:
        if self.schema_version != STOP_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported summary schema version: {self.schema_version}, "
                f"supported version is {STOP_POLICY_SCHEMA_VERSION}"
            )
        if set(self.by_policy) != set(self.policies):
            raise ValueError("by_policy keys must match policies")
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        for policy, summary in self.by_policy.items():
            case_rows = [
                row for row in self.strata if row.policy == policy and row.kind == "case"
            ]
            if sum(row.scheduled for row in case_rows) != summary.scheduled:
                raise ValueError(f"case strata do not partition the scheduled slots of {policy}")
        if self.readiness.split != self.split:
            raise ValueError("summary readiness must describe the summary split")
        return self


__all__ = [
    "AssignedSplit",
    "BASELINE_VARIANT",
    "CaptureStatus",
    "CollectedArticleReference",
    "LaterHopState",
    "LaterHopTerminalOutcome",
    "POST_FOLLOW_REQUIREMENTS",
    "REQUIREMENT_CLASSES",
    "SPLIT_ORDER",
    "STOP_EXPERIMENT_NAME",
    "STOP_POLICY_SCHEMA_VERSION",
    "VARIANT_RECIPES",
    "StopCaptureManifest",
    "StopCaseMacro",
    "StopExclusion",
    "StopExclusionReason",
    "StopHopClass",
    "StopInputTrace",
    "StopLabel",
    "StopLabelRates",
    "StopPairedSummary",
    "StopPlan",
    "StopPolicyCase",
    "StopPolicySummary",
    "StopReadiness",
    "StopRequirement",
    "StopRequirementCount",
    "StopResult",
    "StopReviewSelection",
    "StopRunManifest",
    "StopSplit",
    "StopStrataKind",
    "StopStrataRow",
    "StopSummary",
    "StopTrial",
    "RunPurpose",
    "frozen_state_payload",
    "hop_class_for",
    "later_hop_state_hash",
    "requirement_for",
    "resolve_variant",
    "verify_case_invariants",
]
