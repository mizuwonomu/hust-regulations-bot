"""Contract riêng của sub-experiment `candidate-pair-41-3-position`.

Package sở hữu arm spec, transform trace, lịch, manifest và metric slice của
intervention A. Các record dùng chung như message, execution config, gate
request/result và source reference được import từ `diagnostic_subexp.shared`.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import Field, model_validator

from diagnostic_subexp.shared.contracts import (
    ARTIFACT_ORIGIN,
    ArtifactOrigin,
    DiagnosticExecutionConfig,
    DiagnosticModel,
    DiagnosticSourceIdentity,
    ManifestStatus,
    MessageRecord,
    MetadataMap,
    NonBlank,
    NonEmptyArticleList,
    NonNegativeInt,
    PolicyName,
    PositiveInt,
    RatioMetric,
    SourceReference,
    sha256_text,
    structured_hash,
)

DiagnosticKind: TypeAlias = Literal["candidate-relative-position"]
DiagnosticExclusionReason: TypeAlias = Literal["not_selected_case"]

DIAGNOSTIC_KIND: DiagnosticKind = "candidate-relative-position"
PAIR_ARTICLES: tuple[int, int] = (3, 41)
ORIGINAL_PRESENTED_ORDER: list[int] = [41, 3, 40, 8, 43]
PAIR_ORIENTATIONS = ("3-41", "41-3")


def derive_pair_metadata(presented: list[int]) -> tuple[list[int], str | None]:
    """Suy ra pair slots 1-based và orientation của cặp 3/41 từ presented order."""
    if not isinstance(presented, list):
        raise ValueError("presented candidates must be a list")
    positions = sorted(
        presented.index(article) + 1 for article in PAIR_ARTICLES if article in presented
    )
    if len(positions) != 2:
        raise ValueError(f"presented candidates must contain both articles {PAIR_ARTICLES}")
    orientation = "3-41" if presented.index(3) < presented.index(41) else "41-3"
    return positions, orientation


class CandidateArmSpec(DiagnosticModel):
    """Một arm A với presented order và pair metadata đã khai báo."""

    arm_id: NonBlank
    presented_candidates: NonEmptyArticleList
    matched_pair_id: NonBlank | None = None
    relative_pair_order: str | None = None
    pair_positions: list[PositiveInt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _pair_metadata_invariant(self) -> CandidateArmSpec:
        wants_pair = self.matched_pair_id is not None
        if wants_pair and not self.pair_positions:
            raise ValueError("matched candidate arms require declared pair positions")
        if self.relative_pair_order is not None and not self.pair_positions:
            raise ValueError("relative_pair_order requires pair_positions")
        if self.pair_positions:
            if len(set(self.pair_positions)) != len(self.pair_positions):
                raise ValueError("pair_positions must not contain duplicates")
            derived_positions, derived_order = derive_pair_metadata(self.presented_candidates)
            if self.pair_positions != derived_positions:
                raise ValueError(
                    f"pair_positions {self.pair_positions} do not match the presented order "
                    f"{derived_positions}"
                )
            if self.relative_pair_order != derived_order:
                raise ValueError(
                    f"relative_pair_order {self.relative_pair_order!r} does not match "
                    f"{derived_order!r}"
                )
        if self.relative_pair_order is not None and self.relative_pair_order not in PAIR_ORIENTATIONS:
            raise ValueError(f"relative_pair_order must be one of {PAIR_ORIENTATIONS}")
        return self


class CandidateBatchSpec(DiagnosticModel):
    """Một batch chạy với arm order cố định."""

    batch_id: NonBlank
    arm_order: list[NonBlank]

    @model_validator(mode="after")
    def _arm_order_invariant(self) -> CandidateBatchSpec:
        if not self.arm_order:
            raise ValueError("arm_order must not be empty")
        if len(set(self.arm_order)) != len(self.arm_order):
            raise ValueError("arm_order must not contain duplicates")
        return self


class CandidateSuiteSpec(DiagnosticModel):
    """Spec A dựng trong bộ nhớ, không chứa hash nguồn hay output của model."""

    schema_version: int
    suite_id: NonBlank
    diagnostic_kind: DiagnosticKind
    case_path: NonBlank
    snapshot_path: NonBlank
    inventory_path: NonBlank
    source_case_identity: DiagnosticSourceIdentity
    execution_config: DiagnosticExecutionConfig
    grammar_candidates: NonEmptyArticleList
    arms: list[CandidateArmSpec]
    batches: list[CandidateBatchSpec]
    repeats_per_input_per_batch: PositiveInt

    @model_validator(mode="after")
    def _suite_invariant(self) -> CandidateSuiteSpec:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported diagnostic schema version: {self.schema_version}, supported version is 1"
            )
        if not self.arms:
            raise ValueError("a diagnostic suite requires at least one arm")
        arm_ids = [arm.arm_id for arm in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("arms must have unique arm_id values")
        for arm in self.arms:
            if sorted(arm.presented_candidates) != sorted(self.grammar_candidates):
                raise ValueError(f"arm {arm.arm_id} changes candidate membership")
        if not self.batches:
            raise ValueError("a diagnostic suite requires at least one batch")
        batch_ids = [batch.batch_id for batch in self.batches]
        if len(set(batch_ids)) != len(batch_ids):
            raise ValueError("batches must have unique batch_id values")
        if len(self.batches) != 2:
            raise ValueError("the fixed diagnostic requires exactly two execution batches")
        if self.batches[0].arm_order != arm_ids:
            raise ValueError("batch 0 must follow the declared arm order")
        if self.batches[1].arm_order != list(reversed(arm_ids)):
            raise ValueError("batch 1 must reverse the declared arm order")
        controls = [arm for arm in self.arms if arm.matched_pair_id is None]
        if len(controls) != 1:
            raise ValueError("a candidate suite requires exactly one original control arm")
        if controls[0].presented_candidates != self.grammar_candidates:
            raise ValueError("the original control must keep the fixed grammar order")
        treatment_pairs: dict[str, list[CandidateArmSpec]] = {}
        for arm in self.arms:
            if arm.matched_pair_id is None:
                continue
            treatment_pairs.setdefault(arm.matched_pair_id, []).append(arm)
        if not treatment_pairs:
            raise ValueError("a candidate suite requires at least one matched pair")
        for pair_id, members in treatment_pairs.items():
            if len(members) != 2:
                raise ValueError(f"matched pair {pair_id} requires exactly two arms")
        return self


class CandidateTransformTrace(DiagnosticModel):
    """Kết quả thuần của transform thứ tự candidate, chưa render request."""

    original_candidates: NonEmptyArticleList
    presented_candidates: NonEmptyArticleList
    pair_positions: list[PositiveInt]
    relative_pair_order: str | None
    original_observation_hash: NonBlank
    transformed_observation_hash: NonBlank
    declared_changed_fields: list[NonBlank]
    verified_unchanged_fields: list[NonBlank]

    @model_validator(mode="after")
    def _field_lists(self) -> CandidateTransformTrace:
        for name in ("declared_changed_fields", "verified_unchanged_fields"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        return self


class CandidateInputTrace(DiagnosticModel):
    """Trace bất biến cho một arm A, không chứa nhãn hay identity treatment C."""

    input_trace_id: NonBlank
    case_id: NonBlank
    source_case_identity: DiagnosticSourceIdentity
    diagnostic_kind: DiagnosticKind
    arm_id: NonBlank
    matched_pair_id: NonBlank | None = None
    pair_positions: list[PositiveInt] = Field(default_factory=list)
    relative_pair_order: str | None = None
    question_hash: NonBlank
    original_candidates: NonEmptyArticleList
    presented_candidates: NonEmptyArticleList
    grammar_candidates: NonEmptyArticleList
    original_observation_hash: NonBlank
    transformed_observation_hash: NonBlank
    effective_messages: list[MessageRecord]
    grammar_text: str
    effective_messages_hash: NonBlank
    grammar_hash: NonBlank
    request_fingerprint: NonBlank
    execution_config_hash: NonBlank
    declared_changed_fields: list[NonBlank]
    verified_unchanged_fields: list[NonBlank]

    @model_validator(mode="after")
    def _trace_invariant(self) -> CandidateInputTrace:
        for name in ("declared_changed_fields", "verified_unchanged_fields"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if sha256_text(self.grammar_text) != self.grammar_hash:
            raise ValueError("grammar hash does not match the stored grammar text")
        if (
            structured_hash([message.model_dump(mode="json") for message in self.effective_messages])
            != self.effective_messages_hash
        ):
            raise ValueError("effective message hash does not match the stored messages")
        if not self.effective_messages:
            raise ValueError("effective_messages must not be empty")
        if self.diagnostic_kind != DIAGNOSTIC_KIND:
            raise ValueError("candidate traces require the candidate diagnostic kind")
        if self.original_observation_hash != self.transformed_observation_hash:
            raise ValueError("candidate traces must keep the observation hash")
        return self


class CandidateTrial(DiagnosticModel):
    """Một slot lịch trình A độc lập policy và nhãn."""

    trial_id: NonBlank
    input_trace_id: NonBlank
    case_id: NonBlank
    diagnostic_kind: DiagnosticKind
    arm_id: NonBlank
    batch_id: NonBlank
    repeat_id: NonNegativeInt
    matched_pair_id: NonBlank | None = None
    pair_positions: list[PositiveInt] = Field(default_factory=list)
    relative_pair_order: str | None = None
    presented_candidates: NonEmptyArticleList
    grammar_candidates: NonEmptyArticleList
    observation_hash: NonBlank
    effective_messages_hash: NonBlank
    grammar_hash: NonBlank
    request_fingerprint: NonBlank


class CandidateExclusion(DiagnosticModel):
    """Case bị loại khỏi suite A cùng lý do xác định."""

    case_id: NonBlank
    reason: DiagnosticExclusionReason


class CandidatePlan(DiagnosticModel):
    """Lịch trình A tất định đã validate cùng trace của từng input duy nhất."""

    suite_id: NonBlank
    diagnostic_kind: DiagnosticKind
    source_case_identity: DiagnosticSourceIdentity
    case_source: SourceReference
    snapshot_source: SourceReference
    inventory_source: SourceReference
    policies: list[PolicyName]
    execution_config: DiagnosticExecutionConfig
    traces: list[CandidateInputTrace]
    trials: list[CandidateTrial]
    exclusions: list[CandidateExclusion]
    expected_trial_count: NonNegativeInt
    expected_policy_result_count: NonNegativeInt

    @model_validator(mode="after")
    def _plan_invariant(self) -> CandidatePlan:
        if not self.policies or len(set(self.policies)) != len(self.policies):
            raise ValueError("policies must be nonempty and unique")
        trace_ids = [trace.input_trace_id for trace in self.traces]
        if len(set(trace_ids)) != len(trace_ids):
            raise ValueError("plan traces must have unique input_trace_id values")
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("plan trials must have unique trial_id values")
        if self.expected_trial_count != len(self.trials):
            raise ValueError("expected_trial_count must match the planned schedule")
        if self.expected_policy_result_count != len(self.trials) * len(self.policies):
            raise ValueError("expected_policy_result_count must match the planned schedule")
        known_traces = set(trace_ids)
        for trial in self.trials:
            if trial.input_trace_id not in known_traces:
                raise ValueError(f"trial {trial.trial_id} references an unknown input trace")
        return self


class CandidateManifest(DiagnosticModel):
    """Manifest schema-v3 của một fresh run A."""

    artifact_origin: ArtifactOrigin
    artifact_schema_version: int
    experiment: Literal["fixed-diagnostic"]
    run_id: NonBlank
    started_at: NonBlank
    ended_at: str | None = None
    status: ManifestStatus
    suite_id: NonBlank
    diagnostic_kind: DiagnosticKind
    case_id: NonBlank
    independent_question_count: PositiveInt
    sources: dict[str, SourceReference]
    execution_revision: NonBlank
    execution_dirty: bool
    executed_module_hashes: dict[str, NonBlank]
    policies: list[PolicyName]
    execution_config: DiagnosticExecutionConfig
    policy_config: dict[str, MetadataMap]
    trials: list[CandidateTrial]
    input_trace_ids: list[NonBlank]
    exclusions: list[CandidateExclusion]
    expected_trial_count: NonNegativeInt
    expected_policy_result_count: NonNegativeInt
    definition_sha256: NonBlank
    experiment_definition: dict[str, Any]

    @model_validator(mode="after")
    def _manifest_invariant(self) -> CandidateManifest:
        if self.artifact_origin != ARTIFACT_ORIGIN:
            raise ValueError(f"fresh runs require artifact_origin={ARTIFACT_ORIGIN!r}")
        if self.artifact_schema_version != 3:
            raise ValueError("fresh diagnostic manifests require artifact schema version 3")
        if set(self.sources) != {"case", "snapshot", "inventory"}:
            raise ValueError("fresh manifests must reference exactly case, snapshot and inventory")
        if structured_hash(self.experiment_definition) != self.definition_sha256:
            raise ValueError("definition hash does not match the embedded experiment definition")
        if not self.executed_module_hashes:
            raise ValueError("fresh manifests require executed module hashes")
        for module_name, digest in self.executed_module_hashes.items():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    f"executed module hash for {module_name} is not a SHA-256 digest"
                )
        if self.independent_question_count != 1:
            raise ValueError("this diagnostic version supports exactly one source question")
        if set(self.policy_config) != set(self.policies):
            raise ValueError("policy_config keys must match policies")
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("manifest trials must have unique trial_id values")
        if self.expected_trial_count != len(self.trials):
            raise ValueError("expected_trial_count must match the saved schedule")
        if self.expected_policy_result_count != len(self.trials) * len(self.policies):
            raise ValueError("expected_policy_result_count must match policies and trials")
        if len(set(self.input_trace_ids)) != len(self.input_trace_ids):
            raise ValueError("input_trace_ids must be unique")
        for trial in self.trials:
            if trial.diagnostic_kind != self.diagnostic_kind:
                raise ValueError(f"trial {trial.trial_id} has a different diagnostic kind")
        return self


class CandidateRatioSummary(DiagnosticModel):
    """Coverage và accuracy của một policy A."""

    scheduled: NonNegativeInt
    valid: NonNegativeInt
    errors: NonNegativeInt
    missing: NonNegativeInt
    selection_accuracy: RatioMetric
    decision_accuracy: RatioMetric

    @model_validator(mode="after")
    def _coverage(self) -> CandidateRatioSummary:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        return self


class CandidateActionCountRow(DiagnosticModel):
    """Đếm action theo arm, batch và repeat cho một policy A."""

    policy: PolicyName
    arm_id: NonBlank
    batch_id: NonBlank | None = None
    repeat_id: NonNegativeInt | None = None
    scheduled: NonNegativeInt
    valid: NonNegativeInt
    errors: NonNegativeInt
    missing: NonNegativeInt
    action_counts: dict[str, NonNegativeInt]

    @model_validator(mode="after")
    def _coverage(self) -> CandidateActionCountRow:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        if sum(self.action_counts.values()) != self.valid:
            raise ValueError("action counts must sum to the valid outcomes")
        return self


class CandidateAccuracyRow(DiagnosticModel):
    """Selection và decision accuracy của một arm A trong một case."""

    policy: PolicyName
    arm_id: NonBlank
    case_id: NonBlank
    scheduled: NonNegativeInt
    valid: NonNegativeInt
    errors: NonNegativeInt
    missing: NonNegativeInt
    selection_accuracy: RatioMetric
    decision_accuracy: RatioMetric

    @model_validator(mode="after")
    def _coverage(self) -> CandidateAccuracyRow:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        return self


class CandidateRepeatConsistencyRow(DiagnosticModel):
    """Repeat consistency trong một exact input của một arm A."""

    policy: PolicyName
    arm_id: NonBlank
    batch_id: NonBlank
    request_fingerprint: NonBlank
    scheduled: NonNegativeInt
    valid: NonNegativeInt
    matching_pairs: NonNegativeInt
    valid_pairs: NonNegativeInt
    consistency: RatioMetric

    @model_validator(mode="after")
    def _pair_invariant(self) -> CandidateRepeatConsistencyRow:
        if self.valid_pairs > self.valid or self.matching_pairs > self.valid_pairs:
            raise ValueError("repeat consistency pair counts must respect coverage")
        return self


class CandidateActionAgreement(DiagnosticModel):
    """Đối chiếu action của hai arm A có khai báo matched pair."""

    comparison_id: NonBlank
    policy: PolicyName
    left_arm: NonBlank
    right_arm: NonBlank
    scheduled: NonNegativeInt
    valid: NonNegativeInt
    incomplete: NonNegativeInt
    agreement: RatioMetric
    switch_count: NonNegativeInt
    action_keys: list[NonBlank]
    transition: dict[str, dict[str, NonNegativeInt]]

    @model_validator(mode="after")
    def _agreement_invariant(self) -> CandidateActionAgreement:
        if self.valid + self.incomplete != self.scheduled:
            raise ValueError("valid and incomplete comparisons must partition scheduled ones")
        return self


class CandidateBatchAgreementRow(DiagnosticModel):
    """Đối chiếu action của cùng input identity giữa hai batch của A."""

    policy: PolicyName
    arm_id: NonBlank
    request_fingerprint: NonBlank
    repeat_id: NonNegativeInt
    scheduled: NonNegativeInt
    valid: NonNegativeInt
    agreement: RatioMetric
    switch_count: NonNegativeInt


class CandidatePairSelection(DiagnosticModel):
    """Đếm article 3/41, lựa chọn earlier-of-pair và vị trí tuyệt đối của A."""

    policy: PolicyName
    arm_id: NonBlank
    article_3_count: NonNegativeInt
    article_41_count: NonNegativeInt
    earlier_of_pair_count: NonNegativeInt
    pair_selection_count: NonNegativeInt
    earlier_of_pair: RatioMetric
    absolute_position_counts: dict[str, NonNegativeInt]
    stop_count: NonNegativeInt
    outside_pair_follow_count: NonNegativeInt


class CandidateOutputs(DiagnosticModel):
    """Output riêng của intervention A."""

    pair_selections: list[CandidatePairSelection]
    matched_swaps: list[CandidateActionAgreement]


class CandidateSummary(DiagnosticModel):
    """Aggregate tất định của A, tính lại từ artifact và không gọi model."""

    artifact_schema_version: int
    experiment: Literal["fixed-diagnostic"]
    run_id: NonBlank
    suite_id: NonBlank
    diagnostic_kind: DiagnosticKind
    case_id: NonBlank
    policies: list[PolicyName]
    independent_question_count: PositiveInt
    scheduled: NonNegativeInt
    valid: NonNegativeInt
    errors: NonNegativeInt
    missing: NonNegativeInt
    by_policy: dict[str, CandidateRatioSummary]
    action_counts: list[CandidateActionCountRow]
    accuracy: list[CandidateAccuracyRow]
    repeat_consistency: list[CandidateRepeatConsistencyRow]
    batch_agreement: list[CandidateBatchAgreementRow]
    candidate_relative_position: CandidateOutputs | None = None
    observation_block_order: None = None
    exclusions: list[CandidateExclusion]

    @model_validator(mode="after")
    def _summary_invariant(self) -> CandidateSummary:
        if self.artifact_schema_version != 3:
            raise ValueError("candidate summaries require artifact schema version 3")
        if self.independent_question_count != 1:
            raise ValueError("this diagnostic version supports exactly one source question")
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        if self.candidate_relative_position is None:
            raise ValueError("candidate summaries require candidate output slices")
        return self


__all__ = [
    "CandidateAccuracyRow",
    "CandidateActionAgreement",
    "CandidateActionCountRow",
    "CandidateArmSpec",
    "CandidateBatchAgreementRow",
    "CandidateBatchSpec",
    "CandidateExclusion",
    "CandidateInputTrace",
    "CandidateManifest",
    "CandidateOutputs",
    "CandidatePairSelection",
    "CandidatePlan",
    "CandidateRatioSummary",
    "CandidateRepeatConsistencyRow",
    "CandidateSuiteSpec",
    "CandidateSummary",
    "CandidateTransformTrace",
    "CandidateTrial",
    "DIAGNOSTIC_KIND",
    "DiagnosticExclusionReason",
    "DiagnosticKind",
    "ORIGINAL_PRESENTED_ORDER",
    "PAIR_ARTICLES",
    "PAIR_ORIENTATIONS",
    "derive_pair_metadata",
]
