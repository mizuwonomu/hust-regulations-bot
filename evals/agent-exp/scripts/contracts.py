"""Contract nghiêm ngặt dùng chung cho citation-gate experiment."""

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from src.rag.agent.schema import Decision

QuestionId: TypeAlias = int | str
ArticleId: TypeAlias = int
PolicyName: TypeAlias = Literal["first", "llm"]
ExpectedAction: TypeAlias = Literal["follow", "stop", "unresolved", "no_candidates"]
LabelStatus: TypeAlias = Literal["draft", "approved"]
PermutationCondition: TypeAlias = Literal["repeat", "candidate-order", "seed-order", "combined"]
PermutationSchedule: TypeAlias = Literal["original", "rotate"]
TrialCondition: TypeAlias = Literal[
    "original", "repeat", "candidate-order", "seed-order", "combined"
]
ExclusionReason: TypeAlias = Literal[
    "draft",
    "unresolved",
    "no_candidates",
    "later_hop_seed_order",
    "empty_seed_contexts",
]
JsonValue: TypeAlias = Any
Metadata: TypeAlias = dict[str, JsonValue]

INITIAL_SELECTION_SCHEMA_VERSION = 1
PERMUTATION_SCHEMA_VERSION = 2
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {INITIAL_SELECTION_SCHEMA_VERSION, PERMUTATION_SCHEMA_VERSION}
)
# Contract giữ boundary label-free giữa case, trial, policy và metric


class ContractModel(BaseModel):
    """Model gốc từ chối field chưa khai báo và coercion ngầm."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )


def _validate_question_id(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("question id must be an integer or string")
    return value


def _validate_article_ids(value: Any, *, field_name: str, allow_empty: bool = True) -> Any:
    if not isinstance(value, (list, set, tuple)):
        raise ValueError(f"{field_name} must be a collection of positive integers")

    seen: set[int] = set()
    for article_id in value:
        if isinstance(article_id, bool) or not isinstance(article_id, int) or article_id <= 0:
            raise ValueError(f"{field_name} contains an invalid article id: {article_id!r}")
        if article_id in seen:
            raise ValueError(f"{field_name} contains duplicate article id: {article_id}")
        seen.add(article_id)

    if not allow_empty and not seen:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _validate_contexts(value: Any) -> Any:
    if not isinstance(value, list):
        raise ValueError("contexts must be a list")
    if any(not isinstance(context, str) for context in value):
        raise ValueError("contexts must contain only strings")
    return value


def _validate_metadata(value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")

    def visit(item: Any, path: str) -> None:
        if item is None or isinstance(item, (bool, int, str)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"metadata contains a non-finite float at {path}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError(f"metadata contains a non-string key at {path}")
                visit(child, f"{path}.{key}")
            return
        raise ValueError(f"metadata contains a non-JSON value at {path}")

    visit(value, "metadata")
    return value


def _validate_nonblank(value: Any, *, field_name: str) -> Any:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class SourceFile(ContractModel):
    """Định danh file nguồn và content hash của nó."""

    path: str
    sha256: str

    @field_validator("path", "sha256")
    @classmethod
    def _required_text(cls, value: str, info) -> str:
        return _validate_nonblank(value, field_name=info.field_name)


class SeedRow(ContractModel):
    """Đóng băng một seed retrieval mà không ghép ID với context."""

    id: QuestionId
    question: str
    contexts: list[str]
    seed_dieu: set[ArticleId]

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, value: Any) -> Any:
        return _validate_question_id(value)

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        return _validate_nonblank(value, field_name="question")

    @field_validator("contexts", mode="before")
    @classmethod
    def _contexts(cls, value: Any) -> Any:
        return _validate_contexts(value)

    @field_validator("seed_dieu", mode="before")
    @classmethod
    def _seed_dieu(cls, value: Any) -> Any:
        return set(_validate_article_ids(value, field_name="seed_dieu"))

    @field_serializer("seed_dieu")
    def _serialize_seed_dieu(self, value: set[int]) -> list[int]:
        return sorted(value)


class SeedSnapshot(ContractModel):
    """Đóng băng seed baseline, provenance và whitelist corpus độc lập."""

    schema_version: int = Field(ge=1)
    snapshot_id: str
    dataset_id: str
    baseline_source: SourceFile
    dataset_source: SourceFile
    whitelist_source: SourceFile
    retrieval_config: Metadata
    unavailable_metadata: dict[str, str]
    internal_dieu: set[ArticleId]
    rows: list[SeedRow]

    @field_validator("snapshot_id", "dataset_id")
    @classmethod
    def _identity_text(cls, value: str, info) -> str:
        return _validate_nonblank(value, field_name=info.field_name)

    @field_validator("retrieval_config")
    @classmethod
    def _retrieval_config(cls, value: Metadata) -> Metadata:
        return _validate_metadata(value)

    @field_validator("unavailable_metadata")
    @classmethod
    def _unavailable_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not isinstance(key, str) or not isinstance(reason, str) or not reason.strip() for key, reason in value.items()):
            raise ValueError("unavailable_metadata must map names to non-empty reasons")
        return value

    @field_validator("internal_dieu", mode="before")
    @classmethod
    def _internal_dieu(cls, value: Any) -> Any:
        return set(
            _validate_article_ids(value, field_name="internal_dieu", allow_empty=False)
        )

    @field_validator("rows")
    @classmethod
    def _rows(cls, value: list[SeedRow]) -> list[SeedRow]:
        identities: set[tuple[type[Any], Any]] = set()
        for row in value:
            identity = (type(row.id), row.id)
            if identity in identities:
                raise ValueError(f"rows contains duplicate question id: {row.id!r}")
            identities.add(identity)
        return value

    @field_serializer("internal_dieu")
    def _serialize_internal_dieu(self, value: set[int]) -> list[int]:
        return sorted(value)


class GateInput(ContractModel):
    """Chiếu case thành input không chứa nhãn cho cả hai policy."""

    question: str
    observation: str
    candidates: list[ArticleId]

    @field_validator("question", "observation")
    @classmethod
    def _text(cls, value: str, info) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a string")
        return value

    @field_validator("candidates", mode="before")
    @classmethod
    def _candidates(cls, value: Any) -> Any:
        return _validate_article_ids(value, field_name="candidates")


class GateCase(ContractModel):
    """Đóng băng state initial-gate và boundary review của người."""

    case_id: str
    dataset_id: str
    question_id: QuestionId
    snapshot_id: str
    snapshot_hash: str
    question: str
    observation: str
    observation_hash: str
    candidates: list[ArticleId]
    source_hop: int = Field(ge=0)
    source_run_id: str | None
    source_policy: str | None
    expected_action: ExpectedAction
    acceptable_dieu: set[ArticleId]
    label_reason: str | None
    label_status: LabelStatus
    split: Literal["dev", "heldout"]
    fewshot_overlap: bool | None

    @field_validator(
        "case_id",
        "dataset_id",
        "snapshot_id",
        "snapshot_hash",
        "observation_hash",
    )
    @classmethod
    def _identity_text(cls, value: str, info) -> str:
        return _validate_nonblank(value, field_name=info.field_name)

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        return _validate_nonblank(value, field_name="question")

    @field_validator("observation")
    @classmethod
    def _observation(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("observation must be a string")
        return value

    @field_validator("question_id", mode="before")
    @classmethod
    def _question_id(cls, value: Any) -> Any:
        return _validate_question_id(value)

    @field_validator("candidates", mode="before")
    @classmethod
    def _candidates(cls, value: Any) -> Any:
        return _validate_article_ids(value, field_name="candidates")

    @field_validator("acceptable_dieu", mode="before")
    @classmethod
    def _acceptable_dieu(cls, value: Any) -> Any:
        return set(_validate_article_ids(value, field_name="acceptable_dieu"))

    @field_validator("label_reason")
    @classmethod
    def _label_reason(cls, value: str | None) -> str | None:
        if value is not None and not isinstance(value, str):
            raise ValueError("label_reason must be a string or null")
        return value

    @model_validator(mode="after")
    def _label_invariants(self) -> GateCase:
        candidate_set = set(self.candidates)
        acceptable_set = set(self.acceptable_dieu)
        if not acceptable_set <= candidate_set:
            raise ValueError("acceptable_dieu must be a subset of candidates")
        if self.expected_action == "follow" and not acceptable_set:
            raise ValueError("follow labels require a non-empty acceptable_dieu")
        if self.expected_action in {"stop", "unresolved", "no_candidates"} and acceptable_set:
            raise ValueError(f"{self.expected_action} labels require empty acceptable_dieu")
        if self.expected_action == "no_candidates" and self.candidates:
            raise ValueError("no_candidates cases must have an empty candidate list")
        if self.expected_action != "no_candidates" and not self.candidates:
            raise ValueError("non-no_candidates cases require candidates")
        if self.label_status == "approved" and self.expected_action in {"follow", "stop"}:
            if self.label_reason is None or not self.label_reason.strip():
                raise ValueError("approved semantic labels require a non-blank label_reason")
        return self

    @field_serializer("acceptable_dieu")
    def _serialize_acceptable_dieu(self, value: set[int]) -> list[int]:
        return sorted(value)


class Exclusion(ContractModel):
    """Giải thích vì sao case bị loại khỏi semantic policy schedule."""

    case_id: str
    reason: ExclusionReason


class ReplaySelection(ContractModel):
    """Tách case đã duyệt được chạy khỏi các exclusion rõ ràng."""

    eligible_cases: list[GateCase]
    exclusions: list[Exclusion]


class Usage(ContractModel):
    """Ghi usage token tùy chọn mà policy client cung cấp."""

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    @field_validator("input_tokens", "output_tokens", "total_tokens")
    @classmethod
    def _nonnegative(cls, value: int | None, info) -> int | None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{info.field_name} must be a non-negative integer or null")
        return value


class TrialError(ContractModel):
    """Phân loại lỗi chạy policy mà không lưu traceback."""

    category: Literal["timeout", "transport", "schema", "invalid_candidate", "unexpected"]
    message: str

    @field_validator("message")
    @classmethod
    def _message(cls, value: str) -> str:
        return _validate_nonblank(value, field_name="message")


class DecisionOutcome(ContractModel):
    """Biểu diễn decision hợp lệ hoặc một lỗi đã phân loại."""

    status: Literal["ok", "error"]
    decision: Decision | None
    error: TrialError | None
    latency_ms: float = Field(ge=0)
    usage: Usage | None

    @field_validator("latency_ms", mode="before")
    @classmethod
    def _finite_latency(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("latency_ms must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("latency_ms must be finite")
        return value

    @model_validator(mode="after")
    def _state_invariant(self) -> DecisionOutcome:
        if self.status == "ok" and (self.decision is None or self.error is not None):
            raise ValueError("ok outcome requires a decision and no error")
        if self.status == "error" and (self.decision is not None or self.error is None):
            raise ValueError("error outcome requires an error and no decision")
        return self


class Trial(ContractModel):
    """Định danh một cấu hình input replay chung, độc lập policy và nhãn."""

    trial_id: str
    case_id: str
    repeat_id: int = Field(ge=0)
    condition: TrialCondition
    permutation_id: str
    seed_order: list[int]
    candidate_order: list[ArticleId]
    observation_hash: str

    @field_validator("trial_id", "case_id", "permutation_id", "observation_hash")
    @classmethod
    def _identity_text(cls, value: str, info) -> str:
        return _validate_nonblank(value, field_name=info.field_name)

    @field_validator("repeat_id")
    @classmethod
    def _repeat_id(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("repeat_id must be a non-negative integer")
        return value

    @field_validator("seed_order")
    @classmethod
    def _seed_order(cls, value: list[int]) -> list[int]:
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in value):
            raise ValueError("seed_order must contain non-negative integers")
        if len(set(value)) != len(value):
            raise ValueError("seed_order must not contain duplicates")
        return value

    @field_validator("candidate_order", mode="before")
    @classmethod
    def _candidate_order(cls, value: Any) -> Any:
        return _validate_article_ids(value, field_name="candidate_order")


class ResultRecord(ContractModel):
    """Lưu outcome compact đã chấm điểm của một policy trial."""

    run_id: str
    policy: PolicyName
    trial: Trial
    source_hop: int = Field(ge=0)
    expected_action: ExpectedAction
    acceptable_dieu: set[ArticleId]
    outcome: DecisionOutcome
    selected_position: int | None
    selection_correct: bool | None
    decision_correct: bool

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        return _validate_nonblank(value, field_name="run_id")

    @field_validator("acceptable_dieu", mode="before")
    @classmethod
    def _acceptable_dieu(cls, value: Any) -> Any:
        return set(_validate_article_ids(value, field_name="acceptable_dieu"))

    @field_validator("selected_position")
    @classmethod
    def _selected_position(cls, value: int | None) -> int | None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError("selected_position must be a positive integer or null")
        return value

    @model_validator(mode="after")
    def _result_invariant(self) -> ResultRecord:
        if self.outcome.status == "error":
            if self.selected_position is not None:
                raise ValueError("error results cannot have a selected position")
            if self.expected_action == "follow" and self.selection_correct is not False:
                raise ValueError("follow errors must have selection_correct=False")
            if self.expected_action != "follow" and self.selection_correct is not None:
                raise ValueError("selection_correct is null outside follow labels")
            if self.decision_correct is not False:
                raise ValueError("errors must have decision_correct=False")
            return self

        assert self.outcome.decision is not None
        decision = self.outcome.decision
        if decision.stop:
            if self.selected_position is not None:
                raise ValueError("STOP results cannot have a selected position")
            if self.expected_action == "stop":
                if self.selection_correct is not None:
                    raise ValueError("stop-labeled STOP results must have null selection correctness")
            elif self.selection_correct is not False:
                raise ValueError("STOP on a follow label must have selection_correct=False")
            expected_correct = self.expected_action == "stop"
            if self.decision_correct is not expected_correct:
                raise ValueError("decision_correct does not match the STOP label")
            return self

        if decision.dieu is None or decision.dieu not in self.trial.candidate_order:
            raise ValueError("follow results must select an article in candidate_order")
        expected_position = self.trial.candidate_order.index(decision.dieu) + 1
        if self.selected_position != expected_position:
            raise ValueError("selected_position does not match the selected article")
        if self.expected_action == "follow":
            expected_correct = decision.dieu in self.acceptable_dieu
            if self.selection_correct is not expected_correct:
                raise ValueError("selection_correct does not match the acceptable set")
            if self.decision_correct is not expected_correct:
                raise ValueError("decision_correct does not match the acceptable set")
        else:
            if self.selection_correct is not None or self.decision_correct:
                raise ValueError("follow output is incorrect for a stop label")
        return self

    @field_serializer("acceptable_dieu")
    def _serialize_acceptable_dieu(self, value: set[int]) -> list[int]:
        return sorted(value)


class RatioMetric(ContractModel):
    """Lưu tử số, mẫu số và ratio có thể null của metric."""

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None

    @field_validator("numerator", "denominator")
    @classmethod
    def _integer_counts(cls, value: int, info) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def _finite_value(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("metric value must be numeric or null")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("metric value must be finite or null")
        return value

    @model_validator(mode="after")
    def _denominator_invariant(self) -> RatioMetric:
        if self.denominator == 0 and self.value is not None:
            raise ValueError("zero-denominator metrics must have null value")
        if self.numerator > self.denominator:
            raise ValueError("metric numerator cannot exceed denominator")
        return self


class PolicySummary(ContractModel):
    """Tổng hợp coverage, accuracy, behavioral rate và telemetry theo policy."""

    scheduled: int = Field(ge=0)
    valid: int = Field(ge=0)
    errors: int = Field(ge=0)
    missing: int = Field(ge=0)
    selection_accuracy: RatioMetric
    decision_accuracy: RatioMetric
    first_position_selection_rate: RatioMetric
    costs: Metadata

    @field_validator("scheduled", "valid", "errors", "missing")
    @classmethod
    def _counts(cls, value: int, info) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return value

    @field_validator("costs")
    @classmethod
    def _costs(cls, value: Metadata) -> Metadata:
        return _validate_metadata(value)

    @model_validator(mode="after")
    def _coverage_invariant(self) -> PolicySummary:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        return self


class PairedSummary(ContractModel):
    """Tổng hợp coverage, agreement và correctness của policy pair."""

    scheduled_pairs: int = Field(ge=0)
    valid_pairs: int = Field(ge=0)
    incomplete_pairs: int = Field(ge=0)
    agreement: RatioMetric
    first_wins: int = Field(ge=0)
    llm_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    different_trial_ids: list[str]

    @model_validator(mode="after")
    def _paired_invariant(self) -> PairedSummary:
        if self.valid_pairs + self.incomplete_pairs != self.scheduled_pairs:
            raise ValueError("valid_pairs and incomplete_pairs must partition scheduled_pairs")
        if self.first_wins + self.llm_wins + self.ties != self.valid_pairs:
            raise ValueError("paired correctness counts must partition valid_pairs")
        if len(set(self.different_trial_ids)) != len(self.different_trial_ids):
            raise ValueError("different_trial_ids must be unique")
        return self


class Summary(ContractModel):
    """Lưu aggregate deterministic và exclusion rõ ràng của một run."""

    by_policy: dict[PolicyName, PolicySummary]
    by_candidate_group: dict[str, dict[PolicyName, PolicySummary]]
    by_position: dict[str, dict[PolicyName, RatioMetric]] = Field(default_factory=dict)
    paired: PairedSummary | None
    exclusions: list[Exclusion]


class PermutationConfig(ContractModel):
    """Cấu hình một permutation condition và lịch repeat của nó."""

    condition: PermutationCondition
    schedule: PermutationSchedule
    repeats: int

    @field_validator("repeats", mode="before")
    @classmethod
    def _repeats(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("repeats must be a positive integer excluding booleans")
        return value

    @model_validator(mode="after")
    def _schedule_invariant(self) -> PermutationConfig:
        if self.condition == "repeat" and self.schedule != "original":
            raise ValueError("repeat requires the original schedule")
        if self.condition != "repeat" and self.schedule != "rotate":
            raise ValueError(f"{self.condition} requires the rotate schedule")
        return self


class PermutationPlan(ContractModel):
    """Lưu schedule order đã validate trước khi chạy policy."""

    config: PermutationConfig
    trials: list[Trial]
    exclusions: list[Exclusion]
    expected_trial_count: int
    expected_policy_result_count: int

    @field_validator("expected_trial_count", "expected_policy_result_count")
    @classmethod
    def _counts(cls, value: int, info) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return value

    @model_validator(mode="after")
    def _schedule_invariant(self) -> PermutationPlan:
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("plan trials must have unique trial_id values")
        if self.expected_trial_count != len(self.trials):
            raise ValueError("expected_trial_count must match the planned schedule")
        if any(trial.condition != self.config.condition for trial in self.trials):
            raise ValueError("plan trials must match the configured condition")
        if not self.trials:
            if self.expected_policy_result_count != 0:
                raise ValueError("an empty plan cannot expect policy results")
            return self
        if self.expected_policy_result_count % self.expected_trial_count != 0:
            raise ValueError("expected policy results must be a whole multiple of the trials")
        return self


class QuestionKey(ContractModel):
    """Định danh question gốc trong dataset mà không gộp kiểu ID."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    dataset_id: str
    question_id: QuestionId

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id(cls, value: str) -> str:
        return _validate_nonblank(value, field_name="dataset_id")

    @field_validator("question_id", mode="before")
    @classmethod
    def _question_id(cls, value: Any) -> Any:
        return _validate_question_id(value)


class ActionKey(ContractModel):
    """Định danh STOP hoặc FOLLOW để so sánh, không dùng vị trí."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    action: Literal["stop", "follow"]
    dieu: ArticleId | None

    @model_validator(mode="after")
    def _action_invariant(self) -> ActionKey:
        if self.action == "stop" and self.dieu is not None:
            raise ValueError("STOP action keys cannot carry an article id")
        if self.action == "follow" and (self.dieu is None or self.dieu <= 0):
            raise ValueError("FOLLOW action keys require a positive article id")
        return self


class MeanMetric(ContractModel):
    """Lưu hierarchical mean cùng số defined, eligible và excluded."""

    total: float
    defined: int
    eligible: int
    excluded: int
    value: float | None

    @field_validator("total", mode="before")
    @classmethod
    def _finite_total(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("mean total must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("mean total must be finite")
        return value

    @field_validator("defined", "eligible", "excluded")
    @classmethod
    def _component_counts(cls, value: int, info) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def _finite_value(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("mean value must be numeric or null")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("mean value must be finite or null")
        return value

    @model_validator(mode="after")
    def _mean_invariant(self) -> MeanMetric:
        if self.defined + self.excluded != self.eligible:
            raise ValueError("defined and excluded components must partition eligible ones")
        if self.defined == 0:
            if self.value is not None or self.total != 0.0:
                raise ValueError("a mean without defined components must have null value and zero total")
            return self
        if self.value is None or not math.isclose(
            self.value, self.total / self.defined, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError("mean value does not match its total and defined count")
        return self


class ConsistencySummary(ContractModel):
    """Báo pair agreement pooled cùng case mean và coverage phân cấp."""

    matching_pairs: int
    scheduled_pairs: int
    valid_pairs: int
    pooled: RatioMetric
    eligible_groups: int
    excluded_groups: int
    eligible_cases: int
    excluded_cases: int
    case_mean: MeanMetric

    @field_validator(
        "matching_pairs", "scheduled_pairs", "valid_pairs", "eligible_groups",
        "excluded_groups", "eligible_cases", "excluded_cases",
    )
    @classmethod
    def _counts(cls, value: int, info) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return value

    @model_validator(mode="after")
    def _pair_invariant(self) -> ConsistencySummary:
        if self.matching_pairs > self.valid_pairs or self.valid_pairs > self.scheduled_pairs:
            raise ValueError("pair counts must not exceed their coverage")
        if self.pooled.numerator != self.matching_pairs or self.pooled.denominator != self.valid_pairs:
            raise ValueError("pooled ratio must describe the matching and valid pairs")
        return self


class CaseMetrics(ContractModel):
    """Tổng hợp một case trong một policy và điều kiện permutation."""

    case_id: str
    source_hop: int = Field(ge=0)
    scheduled: int
    valid: int
    errors: int
    missing: int
    selection_accuracy: RatioMetric
    decision_accuracy: RatioMetric
    first_position_selection_rate: RatioMetric
    permutation_consistency: MeanMetric
    repeat_consistency: MeanMetric

    @field_validator("scheduled", "valid", "errors", "missing")
    @classmethod
    def _counts(cls, value: int, info) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return value

    @model_validator(mode="after")
    def _coverage_invariant(self) -> CaseMetrics:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        return self


class QuestionHopMetrics(ContractModel):
    """Giữ mean của case theo một question và source hop."""

    question_key: QuestionKey
    source_hop: int = Field(ge=0)
    scheduled_cases: int
    defined_cases: int
    selection_accuracy: MeanMetric
    decision_accuracy: MeanMetric
    permutation_consistency: MeanMetric
    repeat_consistency: MeanMetric

    @field_validator("scheduled_cases", "defined_cases")
    @classmethod
    def _counts(cls, value: int, info) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return value


class QuestionMacroSummary(ContractModel):
    """Lấy mean case trong từng question rồi lấy mean giữa các question."""

    question_count: int
    selection_accuracy: MeanMetric
    decision_accuracy: MeanMetric
    permutation_consistency: MeanMetric
    repeat_consistency: MeanMetric
    hops: list[QuestionHopMetrics]

    @field_validator("question_count")
    @classmethod
    def _count(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("question_count must be a non-negative integer")
        return value


class GoldPositionCase(ContractModel):
    """Báo accuracy của một case trong một strata gold-position."""

    case_id: str
    correct: int
    scheduled: int
    accuracy: RatioMetric


class GoldPositionGroup(ContractModel):
    """Tách accuracy single-gold theo vị trí và nhóm multi-gold riêng."""

    view: Literal["single_gold", "multi_gold"]
    candidate_count: int
    gold_positions: list[int]
    correct: int
    scheduled: int
    accuracy: RatioMetric
    case_ids: list[str]
    cases: list[GoldPositionCase]
    case_macro: MeanMetric


class PositionCohort(ContractModel):
    """Cho biết tập case được schedule ở mọi candidate position so sánh."""

    candidate_count: int
    positions: list[int]
    case_ids: list[str]
    scheduled_per_position: int


class PositionDistribution(ContractModel):
    """Đếm output hợp lệ theo vị trí và STOP, kèm coverage trên lịch đã chạy."""

    condition: PermutationCondition
    candidate_count: int | None
    scheduled: int
    valid: int
    coverage: RatioMetric
    positions: dict[str, RatioMetric]
    stop: RatioMetric


class IdenticalInputGroup(ContractModel):
    """Nhóm các order khác nhau nhưng tạo cùng full input cho policy."""

    case_id: str
    condition: PermutationCondition
    observation_hash: str
    candidate_order: list[ArticleId]
    permutation_ids: list[str]


class ConditionSummary(ContractModel):
    """Tổng hợp một policy trong một permutation condition."""

    scheduled: int
    valid: int
    errors: int
    missing: int
    selection_accuracy: RatioMetric
    decision_accuracy: RatioMetric
    first_position_selection_rate: RatioMetric
    selection_case_macro: MeanMetric
    decision_case_macro: MeanMetric
    question_macro: QuestionMacroSummary
    cases: list[CaseMetrics]
    permutation_consistency: ConsistencySummary
    repeat_consistency: ConsistencySummary
    gold_positions: list[GoldPositionGroup]
    position_cohorts: list[PositionCohort]
    position_distributions: list[PositionDistribution]
    identical_inputs: list[IdenticalInputGroup]

    @model_validator(mode="after")
    def _coverage_invariant(self) -> ConditionSummary:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        return self


class PolicyPermutationSummary(ContractModel):
    """Tổng hợp một policy qua mọi permutation condition đã chạy."""

    scheduled: int
    valid: int
    errors: int
    missing: int
    telemetry: Metadata
    by_condition: dict[PermutationCondition, ConditionSummary]

    @field_validator("telemetry")
    @classmethod
    def _telemetry(cls, value: Metadata) -> Metadata:
        return _validate_metadata(value)

    @model_validator(mode="after")
    def _coverage_invariant(self) -> PolicyPermutationSummary:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        return self


class PairedConditionSummary(ContractModel):
    """So sánh hai policy trên trial ID chung trong một condition."""

    condition: PermutationCondition
    scheduled_pairs: int
    valid_pairs: int
    coverage: RatioMetric
    agreement: RatioMetric
    a_wins: int
    b_wins: int
    ties: int

    @model_validator(mode="after")
    def _pair_invariant(self) -> PairedConditionSummary:
        if self.valid_pairs > self.scheduled_pairs:
            raise ValueError("valid pairs must not exceed scheduled pairs")
        if self.a_wins + self.b_wins + self.ties != self.valid_pairs:
            raise ValueError("paired correctness counts must partition valid pairs")
        return self


class PairedPolicySummary(ContractModel):
    """Tổng hợp agreement và correctness của policy pair qua các condition."""

    policies: list[PolicyName]
    scheduled_pairs: int
    valid_pairs: int
    coverage: RatioMetric
    agreement: RatioMetric
    a_wins: int
    b_wins: int
    ties: int
    by_condition: list[PairedConditionSummary]

    @model_validator(mode="after")
    def _pair_invariant(self) -> PairedPolicySummary:
        if len(self.policies) != 2 or len(set(self.policies)) != 2:
            raise ValueError("paired policy summaries require two distinct policies")
        if self.valid_pairs > self.scheduled_pairs:
            raise ValueError("valid pairs must not exceed scheduled pairs")
        if self.a_wins + self.b_wins + self.ties != self.valid_pairs:
            raise ValueError("paired correctness counts must partition valid pairs")
        return self


class PermutationSummary(ContractModel):
    """Lưu aggregate permutation deterministic của một run đã validate."""

    schema_version: int = Field(ge=1)
    experiment: Literal["permutation"]
    run_id: str
    by_policy: dict[PolicyName, PolicyPermutationSummary]
    paired: list[PairedPolicySummary]
    exclusions: list[Exclusion]
    schedule: Metadata

    @field_validator("schedule")
    @classmethod
    def _schedule(cls, value: Metadata) -> Metadata:
        return _validate_metadata(value)

    @model_validator(mode="after")
    def _policy_invariant(self) -> PermutationSummary:
        for policy, summary in self.by_policy.items():
            condition_scheduled = sum(item.scheduled for item in summary.by_condition.values())
            if condition_scheduled != summary.scheduled:
                raise ValueError(f"condition counts do not partition policy {policy}")
        return self


class RunManifest(ContractModel):
    """Lưu schedule đầy đủ và provenance cần để audit một run."""

    schema_version: int = Field(ge=1)
    run_id: str
    experiment: Literal["initial-selection", "permutation"]
    started_at: str
    ended_at: str | None
    status: Literal["running", "complete", "completed_with_errors", "incomplete"]
    policies: list[PolicyName]
    trials: list[Trial]
    cases_source: SourceFile
    snapshot_source: SourceFile
    exclusions: list[Exclusion]
    provenance: Metadata
    policy_config: dict[PolicyName, Metadata]
    permutation: PermutationConfig | None = Field(
        default=None,
        validation_alias=AliasChoices("permutation", "permutation_config"),
    )
    expected_trial_count: int | None = None
    expected_policy_result_count: int | None = None
    spec_source: SourceFile | None = None

    @field_validator("run_id", "started_at")
    @classmethod
    def _manifest_text(cls, value: str, info) -> str:
        return _validate_nonblank(value, field_name=info.field_name)

    @field_validator("policies")
    @classmethod
    def _policies(cls, value: list[PolicyName]) -> list[PolicyName]:
        if not value:
            raise ValueError("policies must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("policies must not contain duplicates")
        return value

    @field_validator("provenance")
    @classmethod
    def _provenance(cls, value: Metadata) -> Metadata:
        return _validate_metadata(value)

    @field_validator("policy_config")
    @classmethod
    def _policy_config(cls, value: dict[PolicyName, Metadata]) -> dict[PolicyName, Metadata]:
        for config in value.values():
            _validate_metadata(config)
        return value

    @field_validator("expected_trial_count", "expected_policy_result_count")
    @classmethod
    def _expected_counts(cls, value: int | None, info) -> int | None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{info.field_name} must be a non-negative integer or null")
        return value

    @model_validator(mode="after")
    def _trial_invariant(self) -> RunManifest:
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("trials must have unique trial_id values")
        if any(policy not in {"first", "llm"} for policy in self.policies):
            raise ValueError("manifest contains an unsupported policy")
        if set(self.policy_config) != set(self.policies):
            raise ValueError("policy_config keys must match policies")
        if self.schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported manifest schema version: {self.schema_version}, "
                f"supported versions are {sorted(SUPPORTED_MANIFEST_SCHEMA_VERSIONS)}"
            )
        if self.experiment == "permutation":
            if self.permutation is None:
                raise ValueError("permutation manifests require an effective permutation config")
            if self.schema_version != PERMUTATION_SCHEMA_VERSION:
                raise ValueError("permutation manifests require the permutation schema version")
            if any(trial.condition != self.permutation.condition for trial in self.trials):
                raise ValueError("permutation trials must match the configured condition")
            if self.expected_trial_count != len(self.trials):
                raise ValueError("expected_trial_count must match the saved schedule")
            expected_results = len(self.trials) * len(self.policies)
            if self.expected_policy_result_count != expected_results:
                raise ValueError("expected_policy_result_count must match policies and trials")
            return self
        if self.permutation is not None:
            raise ValueError("initial-selection manifests cannot carry a permutation config")
        if self.schema_version != INITIAL_SELECTION_SCHEMA_VERSION:
            raise ValueError("initial-selection manifests require the initial-selection schema version")
        if any(trial.condition != "original" for trial in self.trials):
            raise ValueError("initial-selection trials must stay in the original condition")
        if self.expected_trial_count is not None or self.expected_policy_result_count is not None:
            raise ValueError("initial-selection manifests cannot carry permutation expected counts")
        return self

    @property
    def permutation_config(self) -> PermutationConfig | None:
        """Cung cấp cấu hình permutation hiệu lực qua tên tương thích dễ hiểu."""
        return self.permutation


__all__ = [
    "ActionKey",
    "ArticleId",
    "CaseMetrics",
    "ConditionSummary",
    "ConsistencySummary",
    "DecisionOutcome",
    "ExpectedAction",
    "Exclusion",
    "ExclusionReason",
    "GateCase",
    "GateInput",
    "GoldPositionCase",
    "GoldPositionGroup",
    "IdenticalInputGroup",
    "INITIAL_SELECTION_SCHEMA_VERSION",
    "JsonValue",
    "LabelStatus",
    "MeanMetric",
    "Metadata",
    "PairedConditionSummary",
    "PairedPolicySummary",
    "PairedSummary",
    "PERMUTATION_SCHEMA_VERSION",
    "PermutationCondition",
    "PermutationConfig",
    "PermutationPlan",
    "PermutationSchedule",
    "PermutationSummary",
    "PolicyName",
    "PolicyPermutationSummary",
    "PolicySummary",
    "PositionCohort",
    "PositionDistribution",
    "QuestionHopMetrics",
    "QuestionId",
    "QuestionKey",
    "QuestionMacroSummary",
    "RatioMetric",
    "ReplaySelection",
    "ResultRecord",
    "RunManifest",
    "SeedRow",
    "SeedSnapshot",
    "SourceFile",
    "Summary",
    "Trial",
    "TrialCondition",
    "TrialError",
    "Usage",
    "SUPPORTED_MANIFEST_SCHEMA_VERSIONS",
]
