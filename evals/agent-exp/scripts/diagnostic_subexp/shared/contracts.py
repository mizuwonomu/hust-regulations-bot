"""Contract nền dùng chung cho mọi harness diagnostic chạy offline.

Module chỉ giữ các record thực sự xuất hiện ở nhiều sub-experiment: message
hiệu lực, cấu hình client, gate request/result, source reference, provenance
hash và các kiểu primitive có validate nghiêm ngặt. Contract riêng của từng
sub-experiment nằm trong package của nó.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from contracts import DecisionOutcome, RatioMetric

PolicyName: TypeAlias = Literal["first", "llm"]
ManifestStatus: TypeAlias = Literal[
    "prepared",
    "running",
    "complete",
    "completed_with_errors",
    "incomplete",
]
MessageRole: TypeAlias = Literal["system", "human", "ai"]
Metadata: TypeAlias = dict[str, Any]
JsonValue: TypeAlias = Any
QuestionId: TypeAlias = int | str

ARTIFACT_SCHEMA_VERSION = 3
DIAGNOSTIC_SCHEMA_VERSION = 1
EXPERIMENT_NAME = "fixed-diagnostic"
ARTIFACT_ORIGIN = "post-refactor-fresh-run"
ArtifactOrigin: TypeAlias = Literal["post-refactor-fresh-run"]
PrepareStatus: TypeAlias = Literal["created", "existing-result"]
# Ba result root lịch sử chỉ được kiểm tra bằng Path.exists, không bao giờ đọc con
HISTORICAL_RESULT_ROOTS: tuple[str, ...] = (
    "evals/agent-exp/results/fixed-diagnostic-ab-live-v1/a-candidate-position",
    "evals/agent-exp/results/fixed-diagnostic-ab-live-v1/b-observation-order",
    "evals/agent-exp/results/prompt-ablation/c-llm-v1",
)


@dataclass(frozen=True, slots=True)
class PrepareOutcome:
    """Kết quả prepare: tạo run mới hoặc giữ nguyên result đang tồn tại."""

    status: PrepareStatus
    run_dir: Path
    manifest: Any | None = None
    plan: Any | None = None


class DiagnosticModel(BaseModel):
    """Model gốc từ chối field lạ và mọi coercion ngầm."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )


def _nonblank(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("value must be a non-empty string")
    return value


def _question_id(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("question id must be an integer or string")
    return value


def _article_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid article id: {value!r}")
    return value


def _article_list(value: Any) -> Any:
    if not isinstance(value, (list, tuple, set)):
        raise ValueError("value must be a collection of positive integers")
    seen: set[int] = set()
    for article_id in value:
        checked = _article_id(article_id)
        if checked in seen:
            raise ValueError(f"duplicate article id: {checked}")
        seen.add(checked)
    return list(value)


def _nonempty_article_list(value: Any) -> Any:
    articles = _article_list(value)
    if not articles:
        raise ValueError("value must not be empty")
    return articles


def _nonneg_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"value must be an integer >= 0: {value!r}")
    return value


def _positive_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"value must be an integer >= 1: {value!r}")
    return value


def _span(value: Any) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("value must be a two-element [start, end) span")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end < start
    ):
        raise ValueError("value must be a non-negative end-exclusive span")
    return [start, end]


def _metadata(value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError("value must be an object")

    def visit(item: Any, path: str) -> None:
        if item is None or isinstance(item, (bool, int, str)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"value contains a non-finite float at {path}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError(f"value contains a non-string key at {path}")
                visit(child, f"{path}.{key}")
            return
        raise ValueError(f"value contains a non-JSON value at {path}")

    visit(value, "metadata")
    return value


NonBlank: TypeAlias = Annotated[str, AfterValidator(_nonblank)]
TypedQuestionId: TypeAlias = Annotated[QuestionId, BeforeValidator(_question_id)]
ArticleId: TypeAlias = Annotated[int, AfterValidator(_article_id)]
ArticleList: TypeAlias = Annotated[list[ArticleId], AfterValidator(_article_list)]
NonEmptyArticleList: TypeAlias = Annotated[list[ArticleId], AfterValidator(_nonempty_article_list)]
NonNegativeInt: TypeAlias = Annotated[int, AfterValidator(_nonneg_int)]
PositiveInt: TypeAlias = Annotated[int, AfterValidator(_positive_int)]
ByteSpan: TypeAlias = Annotated[list[int], AfterValidator(_span)]
MetadataMap: TypeAlias = Annotated[dict[str, Any], AfterValidator(_metadata)]


def sha256_bytes(value: bytes) -> str:
    """Băm bytes bằng SHA-256."""
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Băm text UTF-8 bằng SHA-256."""
    return sha256_bytes(value.encode("utf-8"))


def canonical_json(value: Any) -> bytes:
    """Serialize JSON tất định, giữ Unicode và từ chối số không hữu hạn."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def structured_hash(value: Any) -> str:
    """Băm cấu trúc JSON tất định bằng SHA-256."""
    return sha256_bytes(canonical_json(value))


class MessageRecord(DiagnosticModel):
    """Một message hiệu lực đã gửi tới client local."""

    role: MessageRole
    content: str

    @field_validator("content")
    @classmethod
    def _content(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("message content must be a string")
        return value


class DiagnosticSourceIdentity(DiagnosticModel):
    """Định danh nguồn case đã đóng băng, giữ nguyên kiểu của question id."""

    dataset_id: NonBlank
    question_id: TypedQuestionId
    case_id: NonBlank
    source_hop: NonNegativeInt


class DiagnosticExecutionConfig(DiagnosticModel):
    """Cấu hình non-secret đã ghi cho client local của diagnostic."""

    model_alias: NonBlank
    base_url: NonBlank
    temperature: float
    max_completion_tokens: PositiveInt
    enable_thinking: bool
    timeout_seconds: float | None = None
    max_retries: NonNegativeInt | None = None
    extra_request_options: MetadataMap = Field(default_factory=dict)

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("temperature must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("temperature must be finite")
        return float(value)

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _timeout(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("timeout_seconds must be numeric or null")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("timeout_seconds must be positive or null")
        return float(value)


class DiagnosticGateRequest(DiagnosticModel):
    """Request hiệu lực đã render, không chứa nhãn hay kỳ vọng nào."""

    question: str
    observation: str
    presented_candidates: NonEmptyArticleList
    grammar_candidates: NonEmptyArticleList
    effective_messages: list[MessageRecord]
    grammar_text: str

    @field_validator("question", "observation", "grammar_text")
    @classmethod
    def _text(cls, value: str, info) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a string")
        return value

    @field_validator("effective_messages")
    @classmethod
    def _messages(cls, value: list[MessageRecord]) -> list[MessageRecord]:
        if not value:
            raise ValueError("effective_messages must not be empty")
        return value


class SourceReference(DiagnosticModel):
    """Tham chiếu nguồn nền canonical mà fresh reload bắt buộc verify."""

    role: Literal["case", "snapshot", "inventory"]
    path: NonBlank
    sha256: NonBlank
    required_for_replay: bool = True

    @model_validator(mode="after")
    def _canonical_path(self) -> SourceReference:
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source references must be canonical repository-relative paths")
        if not self.required_for_replay:
            raise ValueError("shared base sources must be required for replay")
        return self


class DiagnosticResult(DiagnosticModel):
    """Outcome compact đã chấm điểm của một policy trial."""

    run_id: NonBlank
    trial_id: NonBlank
    policy: PolicyName
    input_trace_id: NonBlank
    request_fingerprint: NonBlank
    outcome: DecisionOutcome
    selected_article: ArticleId | None = None
    selected_position: PositiveInt | None = None
    selection_correct: bool | None = None
    decision_correct: bool


DiagnosticGateRequest.model_rebuild()
SourceReference.model_rebuild()
DiagnosticResult.model_rebuild()

__all__ = [
    "ARTIFACT_ORIGIN",
    "ARTIFACT_SCHEMA_VERSION",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "EXPERIMENT_NAME",
    "HISTORICAL_RESULT_ROOTS",
    "ArticleId",
    "ArticleList",
    "ArtifactOrigin",
    "ByteSpan",
    "DiagnosticExecutionConfig",
    "DiagnosticGateRequest",
    "DiagnosticModel",
    "DiagnosticResult",
    "DiagnosticSourceIdentity",
    "JsonValue",
    "ManifestStatus",
    "MessageRecord",
    "MessageRole",
    "Metadata",
    "MetadataMap",
    "NonBlank",
    "NonEmptyArticleList",
    "NonNegativeInt",
    "PolicyName",
    "PositiveInt",
    "PrepareOutcome",
    "PrepareStatus",
    "QuestionId",
    "RatioMetric",
    "SourceReference",
    "TypedQuestionId",
    "canonical_json",
    "sha256_bytes",
    "sha256_text",
    "structured_hash",
]
