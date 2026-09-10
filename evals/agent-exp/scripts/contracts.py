"""Strict records shared by the initial citation-gate experiment."""

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from src.rag.agent.schema import Decision

QuestionId: TypeAlias = int | str
ArticleId: TypeAlias = int
PolicyName: TypeAlias = Literal["first", "llm"]
ExpectedAction: TypeAlias = Literal["follow", "stop", "unresolved", "no_candidates"]
LabelStatus: TypeAlias = Literal["draft", "approved"]
JsonValue: TypeAlias = Any
Metadata: TypeAlias = dict[str, JsonValue]


class ContractModel(BaseModel):
    """Base model that rejects undeclared fields and implicit scalar coercion."""

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
    """Identify a source file and its content hash."""

    path: str
    sha256: str

    @field_validator("path", "sha256")
    @classmethod
    def _required_text(cls, value: str, info) -> str:
        return _validate_nonblank(value, field_name=info.field_name)


class SeedRow(ContractModel):
    """Freeze one baseline retrieval seed without pairing IDs to contexts."""

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
    """Freeze baseline seeds, provenance, and the independent corpus whitelist."""

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
    """Project a case to the label-free input accepted by either policy."""

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
    """Freeze one initial-gate decision state and its human-review boundary."""

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
    """Explain why a case is absent from semantic policy scheduling."""

    case_id: str
    reason: Literal["draft", "unresolved", "no_candidates"]


class ReplaySelection(ContractModel):
    """Separate approved executable cases from explicit exclusions."""

    eligible_cases: list[GateCase]
    exclusions: list[Exclusion]


class Usage(ContractModel):
    """Record optional token usage exposed by a policy client."""

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
    """Classify one policy execution failure without storing a traceback."""

    category: Literal["timeout", "transport", "schema", "invalid_candidate", "unexpected"]
    message: str

    @field_validator("message")
    @classmethod
    def _message(cls, value: str) -> str:
        return _validate_nonblank(value, field_name="message")


class DecisionOutcome(ContractModel):
    """Represent one valid decision or one classified execution failure."""

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
    """Identify one shared original-order replay input."""

    trial_id: str
    case_id: str
    repeat_id: int = Field(ge=0)
    condition: Literal["original"]
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
    """Persist the compact scored outcome for one policy trial."""

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
    """Store a metric numerator, denominator, and nullable ratio."""

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
    """Aggregate coverage, accuracy, behavioral rate, and telemetry by policy."""

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
    """Aggregate paired policy coverage, agreement, and correctness outcomes."""

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
    """Persist deterministic aggregates and explicit exclusions for one run."""

    by_policy: dict[PolicyName, PolicySummary]
    by_candidate_group: dict[str, dict[PolicyName, PolicySummary]]
    by_position: dict[str, dict[PolicyName, RatioMetric]] = Field(default_factory=dict)
    paired: PairedSummary | None
    exclusions: list[Exclusion]


class RunManifest(ContractModel):
    """Persist the complete schedule and provenance needed to audit a run."""

    schema_version: int = Field(ge=1)
    run_id: str
    experiment: Literal["initial-selection"]
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

    @model_validator(mode="after")
    def _trial_invariant(self) -> RunManifest:
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("trials must have unique trial_id values")
        if any(policy not in {"first", "llm"} for policy in self.policies):
            raise ValueError("manifest contains an unsupported policy")
        if set(self.policy_config) != set(self.policies):
            raise ValueError("policy_config keys must match policies")
        return self


__all__ = [
    "ArticleId",
    "DecisionOutcome",
    "ExpectedAction",
    "Exclusion",
    "GateCase",
    "GateInput",
    "JsonValue",
    "LabelStatus",
    "Metadata",
    "PairedSummary",
    "PolicyName",
    "PolicySummary",
    "QuestionId",
    "RatioMetric",
    "ReplaySelection",
    "ResultRecord",
    "RunManifest",
    "SeedRow",
    "SeedSnapshot",
    "SourceFile",
    "Summary",
    "Trial",
    "TrialError",
    "Usage",
]
