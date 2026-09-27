"""Contract riêng của sub-experiment `article-42-observation-block-order`.

Package sở hữu arm spec, observation block, transform, lịch, manifest và metric
slice của intervention B. Các record dùng chung như message, execution config,
gate request/result và source reference được import từ `diagnostic_subexp.shared`.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from diagnostic_subexp.shared.contracts import (
    ARTIFACT_ORIGIN,
    ArtifactOrigin,
    ArticleList,
    ByteSpan,
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

DiagnosticKind: TypeAlias = Literal["observation-block-order"]
DiagnosticExclusionReason: TypeAlias = Literal["not_selected_case"]

DIAGNOSTIC_KIND: DiagnosticKind = "observation-block-order"
DEFAULT_PAGE_SPLIT_NOTE = (
    "Articles 3 and 41 share b42-foundation-timing, so intervention B cannot "
    "distinguish the order of their mention inside that atomic sentence"
)


class ObservationArmSpec(DiagnosticModel):
    """Một arm B với transform observation đã khai báo."""

    arm_id: NonBlank
    presented_candidates: NonEmptyArticleList
    observation_transform_id: NonBlank
    matched_pair_id: NonBlank | None = None

    @model_validator(mode="after")
    def _arm_invariant(self) -> ObservationArmSpec:
        if self.matched_pair_id is None:
            raise ValueError("observation arms require a matched pair identity")
        return self


class ObservationBlockSpec(DiagnosticModel):
    """Một atomic evidence block trong observation đã đóng băng."""

    block_id: NonBlank
    context_article_id: PositiveInt
    source_line_count: PositiveInt
    source_line_numbers: list[PositiveInt]
    source_byte_span: ByteSpan
    source_text_sha256: NonBlank
    referenced_candidates: ArticleList
    must_remain_contiguous: bool

    @model_validator(mode="after")
    def _block_invariant(self) -> ObservationBlockSpec:
        numbers = self.source_line_numbers
        if len(set(numbers)) != len(numbers):
            raise ValueError("source_line_numbers must not contain duplicates")
        if numbers != sorted(numbers):
            raise ValueError("source_line_numbers must be ordered")
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError("source_line_numbers must be contiguous")
        if self.source_line_count != len(numbers):
            raise ValueError("source_line_count must match the declared source line numbers")
        return self


class ObservationSectionSpec(DiagnosticModel):
    """Một đoạn context không đổi ngoài các atomic block của B."""

    section_id: NonBlank
    source_line_numbers: list[PositiveInt]
    source_byte_span: ByteSpan
    source_text_sha256: NonBlank
    referenced_candidates: ArticleList

    @model_validator(mode="after")
    def _section_invariant(self) -> ObservationSectionSpec:
        numbers = self.source_line_numbers
        if len(set(numbers)) != len(numbers):
            raise ValueError("source_line_numbers must not contain duplicates")
        if numbers != sorted(numbers):
            raise ValueError("source_line_numbers must be ordered")
        return self


class PageSplitGroupSpec(DiagnosticModel):
    """Hai dòng của một câu bị ngắt trang phải giữ nguyên tính atomic."""

    group_id: NonBlank
    block_id: NonBlank
    source_line_numbers: list[PositiveInt]
    line_sha256: list[NonBlank]

    @model_validator(mode="after")
    def _group_invariant(self) -> PageSplitGroupSpec:
        if len(self.line_sha256) != 2:
            raise ValueError("page split groups require exactly two line hashes")
        return self


class ObservationTransformSpec(DiagnosticModel):
    """Ánh xạ thứ tự block cùng prefix/suffix bất biến theo byte."""

    transform_id: NonBlank
    ordered_block_ids: list[NonBlank]
    prefix_byte_span: ByteSpan
    prefix_sha256: NonBlank
    suffix_byte_span: ByteSpan
    suffix_sha256: NonBlank

    @model_validator(mode="after")
    def _transform_invariant(self) -> ObservationTransformSpec:
        if len(set(self.ordered_block_ids)) != len(self.ordered_block_ids):
            raise ValueError("ordered_block_ids must not contain duplicates")
        return self


class ObservationBatchSpec(DiagnosticModel):
    """Một batch chạy với arm order cố định."""

    batch_id: NonBlank
    arm_order: list[NonBlank]

    @model_validator(mode="after")
    def _arm_order_invariant(self) -> ObservationBatchSpec:
        if not self.arm_order:
            raise ValueError("arm_order must not be empty")
        if len(set(self.arm_order)) != len(self.arm_order):
            raise ValueError("arm_order must not contain duplicates")
        return self


class ObservationSuiteSpec(DiagnosticModel):
    """Spec B dựng trong bộ nhớ, không chứa hash nguồn hay output của model."""

    schema_version: int
    suite_id: NonBlank
    diagnostic_kind: DiagnosticKind
    case_path: NonBlank
    snapshot_path: NonBlank
    inventory_path: NonBlank
    source_case_identity: DiagnosticSourceIdentity
    execution_config: DiagnosticExecutionConfig
    grammar_candidates: NonEmptyArticleList
    arms: list[ObservationArmSpec]
    observation_blocks: list[ObservationBlockSpec]
    observation_sections: list[ObservationSectionSpec]
    page_split_group: PageSplitGroupSpec | None = None
    observation_transforms: list[ObservationTransformSpec]
    batches: list[ObservationBatchSpec]
    repeats_per_input_per_batch: PositiveInt

    @model_validator(mode="after")
    def _suite_invariant(self) -> ObservationSuiteSpec:
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
            if arm.presented_candidates != self.grammar_candidates:
                raise ValueError("observation arms must keep the original candidate order")
        batch_ids = [batch.batch_id for batch in self.batches]
        if len(self.batches) != 2:
            raise ValueError("the fixed diagnostic requires exactly two execution batches")
        if len(set(batch_ids)) != len(batch_ids):
            raise ValueError("batches must have unique batch_id values")
        if self.batches[0].arm_order != arm_ids:
            raise ValueError("batch 0 must follow the declared arm order")
        if self.batches[1].arm_order != list(reversed(arm_ids)):
            raise ValueError("batch 1 must reverse the declared arm order")
        if not self.observation_blocks:
            raise ValueError("observation diagnostic suites require atomic blocks")
        block_by_id = {block.block_id: block for block in self.observation_blocks}
        if len(block_by_id) != len(self.observation_blocks):
            raise ValueError("observation_blocks must have unique block_id values")
        spans = [tuple(block.source_byte_span) for block in self.observation_blocks]
        for index, (start, end) in enumerate(spans):
            for other_start, other_end in spans[index + 1 :]:
                if start < other_end and other_start < end:
                    raise ValueError("atomic block byte spans must not overlap")
        transform_ids = [transform.transform_id for transform in self.observation_transforms]
        if len(set(transform_ids)) != len(transform_ids):
            raise ValueError("observation_transforms must have unique transform_id values")
        for transform in self.observation_transforms:
            missing = set(transform.ordered_block_ids) - set(block_by_id)
            if missing:
                raise ValueError(
                    f"transform {transform.transform_id} references unknown blocks "
                    f"{sorted(missing)}"
                )
        pair_ids = {arm.matched_pair_id for arm in self.arms}
        if len(pair_ids) != 1 or len(self.arms) != 2:
            raise ValueError("observation arms must share exactly one matched pair id")
        before_order = [block.block_id for block in self.observation_blocks]
        identity_members = [
            arm
            for arm in self.arms
            if self._transform_order(arm.observation_transform_id) == before_order
        ]
        if len(identity_members) != 1:
            raise ValueError("intervention B requires exactly one identity-order control arm")
        orders = [self._transform_order(arm.observation_transform_id) for arm in self.arms]
        if orders[0] == orders[1]:
            raise ValueError("intervention B requires the two arms to use different block orders")
        if self.page_split_group is not None:
            group = self.page_split_group
            if group.block_id not in block_by_id:
                raise ValueError("page split group must reference a declared block")
            block = block_by_id[group.block_id]
            if block.source_line_numbers != group.source_line_numbers:
                raise ValueError("page split group must match its block source lines")
            if not block.must_remain_contiguous:
                raise ValueError("page split groups require a contiguous block")
        return self

    def _transform_order(self, transform_id: str) -> list[str]:
        for transform in self.observation_transforms:
            if transform.transform_id == transform_id:
                return list(transform.ordered_block_ids)
        raise ValueError(f"unknown observation transform: {transform_id!r}")


class ObservationBlockTrace(DiagnosticModel):
    """Block đã dịch chuyển cùng text, hash và vị trí trước/sau."""

    block_id: NonBlank
    text: str
    text_sha256: NonBlank
    referenced_candidates: ArticleList
    before_position: PositiveInt
    after_position: PositiveInt
    before_byte_span: ByteSpan
    after_byte_span: ByteSpan


class ObservationMovedBlock(DiagnosticModel):
    """Ghi lại vị trí trước/sau của một block đã đổi chỗ."""

    block_id: NonBlank
    before_position: PositiveInt
    after_position: PositiveInt


class ObservationLineOrigin(DiagnosticModel):
    """Dòng output ánh xạ về dòng gốc, segment và hash dòng."""

    output_line_number: PositiveInt
    original_line_number: PositiveInt
    segment_id: NonBlank
    line_sha256: NonBlank


class ObservationSectionTrace(DiagnosticModel):
    """Section không đổi kèm span trước/sau và hash bytes."""

    section_id: NonBlank
    source_byte_span: ByteSpan
    output_byte_span: ByteSpan
    text_sha256: NonBlank
    referenced_candidates: ArticleList


class ObservationPageSplitProof(DiagnosticModel):
    """Bằng chứng hai dòng của câu ngắt trang vẫn atomic sau transform."""

    group_id: NonBlank
    block_id: NonBlank
    line_sha256: list[NonBlank]
    before_line_numbers: list[PositiveInt]
    after_line_numbers: list[PositiveInt]

    @model_validator(mode="after")
    def _proof_invariant(self) -> ObservationPageSplitProof:
        if len(self.line_sha256) != 2:
            raise ValueError("page split proofs require exactly two line hashes")
        return self


class ObservationTransformTrace(DiagnosticModel):
    """Kết quả đầy đủ của transform thứ tự block, chưa render request."""

    transform_id: NonBlank
    original_observation: str
    transformed_observation: str
    before_block_order: list[NonBlank]
    after_block_order: list[NonBlank]
    blocks: list[ObservationBlockTrace]
    moved_blocks: list[ObservationMovedBlock]
    line_origin_map: list[ObservationLineOrigin]
    sections: list[ObservationSectionTrace]
    page_split_group: ObservationPageSplitProof | None = None
    declared_changed_fields: list[NonBlank]
    verified_unchanged_fields: list[NonBlank]

    @model_validator(mode="after")
    def _transform_invariant(self) -> ObservationTransformTrace:
        for name in ("declared_changed_fields", "verified_unchanged_fields"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if self.before_block_order != [block.block_id for block in self.blocks]:
            raise ValueError("before_block_order must match the block records")
        if sorted(self.before_block_order) != sorted(self.after_block_order):
            raise ValueError("both block orders must contain the same blocks")
        if not self.blocks or not self.line_origin_map:
            raise ValueError("observation transforms require block and line-origin evidence")
        return self


class ObservationInputTrace(DiagnosticModel):
    """Trace bất biến cho một arm B, không chứa nhãn hay identity treatment C.

    Pair metadata của Article 3/41 được ghi lại như bằng chứng và phải khớp
    presented order.
    """

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
    original_observation: str | None = None
    transformed_observation: str | None = None
    before_block_order: list[NonBlank] = Field(default_factory=list)
    after_block_order: list[NonBlank] = Field(default_factory=list)
    blocks: list[ObservationBlockTrace] = Field(default_factory=list)
    moved_blocks: list[ObservationMovedBlock] = Field(default_factory=list)
    line_origin_map: list[ObservationLineOrigin] = Field(default_factory=list)
    source_and_output_byte_spans: MetadataMap = Field(default_factory=dict)
    unchanged_context_sections: list[ObservationSectionTrace] = Field(default_factory=list)
    page_split_group: ObservationPageSplitProof | None = None

    @model_validator(mode="after")
    def _trace_invariant(self) -> ObservationInputTrace:
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
            raise ValueError("observation traces require the observation diagnostic kind")
        if self.original_observation is None or self.transformed_observation is None:
            raise ValueError("observation traces require complete before and after observations")
        if sha256_text(self.original_observation) != self.original_observation_hash:
            raise ValueError("original observation hash does not match its text")
        if sha256_text(self.transformed_observation) != self.transformed_observation_hash:
            raise ValueError("transformed observation hash does not match its text")
        if not self.blocks or not self.line_origin_map:
            raise ValueError("observation traces require block and line-origin evidence")
        if self.before_block_order != [block.block_id for block in self.blocks]:
            raise ValueError("before_block_order must match the block records")
        return self


class ObservationTrial(DiagnosticModel):
    """Một slot lịch trình B độc lập policy và nhãn."""

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


class ObservationExclusion(DiagnosticModel):
    """Case bị loại khỏi suite B cùng lý do xác định."""

    case_id: NonBlank
    reason: DiagnosticExclusionReason


class ObservationPlan(DiagnosticModel):
    """Lịch trình B tất định đã validate cùng trace của từng input duy nhất."""

    suite_id: NonBlank
    diagnostic_kind: DiagnosticKind
    source_case_identity: DiagnosticSourceIdentity
    case_source: SourceReference
    snapshot_source: SourceReference
    inventory_source: SourceReference
    policies: list[PolicyName]
    execution_config: DiagnosticExecutionConfig
    traces: list[ObservationInputTrace]
    trials: list[ObservationTrial]
    exclusions: list[ObservationExclusion]
    expected_trial_count: NonNegativeInt
    expected_policy_result_count: NonNegativeInt

    @model_validator(mode="after")
    def _plan_invariant(self) -> ObservationPlan:
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


class ObservationManifest(DiagnosticModel):
    """Manifest schema-v3 của một fresh run B."""

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
    trials: list[ObservationTrial]
    input_trace_ids: list[NonBlank]
    exclusions: list[ObservationExclusion]
    expected_trial_count: NonNegativeInt
    expected_policy_result_count: NonNegativeInt
    definition_sha256: NonBlank
    experiment_definition: dict[str, object]

    @model_validator(mode="after")
    def _manifest_invariant(self) -> ObservationManifest:
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


class ObservationRatioSummary(DiagnosticModel):
    """Coverage và accuracy của một policy B."""

    scheduled: NonNegativeInt
    valid: NonNegativeInt
    errors: NonNegativeInt
    missing: NonNegativeInt
    selection_accuracy: RatioMetric
    decision_accuracy: RatioMetric

    @model_validator(mode="after")
    def _coverage(self) -> ObservationRatioSummary:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        return self


class ObservationActionCountRow(DiagnosticModel):
    """Đếm action theo arm, batch và repeat cho một policy B."""

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
    def _coverage(self) -> ObservationActionCountRow:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        if sum(self.action_counts.values()) != self.valid:
            raise ValueError("action counts must sum to the valid outcomes")
        return self


class ObservationAccuracyRow(DiagnosticModel):
    """Selection và decision accuracy của một arm B trong một case."""

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
    def _coverage(self) -> ObservationAccuracyRow:
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        return self


class ObservationRepeatConsistencyRow(DiagnosticModel):
    """Repeat consistency trong một exact input của một arm B."""

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
    def _pair_invariant(self) -> ObservationRepeatConsistencyRow:
        if self.valid_pairs > self.valid or self.matching_pairs > self.valid_pairs:
            raise ValueError("repeat consistency pair counts must respect coverage")
        return self


class ObservationActionAgreement(DiagnosticModel):
    """Đối chiếu action của hai arm B có khai báo matched pair."""

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
    def _agreement_invariant(self) -> ObservationActionAgreement:
        if self.valid + self.incomplete != self.scheduled:
            raise ValueError("valid and incomplete comparisons must partition scheduled ones")
        return self


class ObservationBatchAgreementRow(DiagnosticModel):
    """Đối chiếu action của cùng input identity giữa hai batch của B."""

    policy: PolicyName
    arm_id: NonBlank
    request_fingerprint: NonBlank
    repeat_id: NonNegativeInt
    scheduled: NonNegativeInt
    valid: NonNegativeInt
    agreement: RatioMetric
    switch_count: NonNegativeInt


class ObservationSelectedBlockRow(DiagnosticModel):
    """Block hỗ trợ của article được chọn, kèm vị trí 1-based sau transform."""

    policy: PolicyName
    arm_id: NonBlank
    batch_id: NonBlank
    repeat_id: NonNegativeInt
    selected_article: PositiveInt
    block_ids: list[NonBlank]
    block_positions: list[PositiveInt]


class ObservationOutputs(DiagnosticModel):
    """Output riêng của intervention B."""

    transitions: list[ObservationActionAgreement]
    selected_blocks: list[ObservationSelectedBlockRow]
    shared_block_note: NonBlank
    observation_transform_trace_ids: dict[str, NonBlank]

    @model_validator(mode="after")
    def _outputs_invariant(self) -> ObservationOutputs:
        if not self.transitions:
            raise ValueError("observation outputs require at least one comparison")
        return self


class ObservationSummary(DiagnosticModel):
    """Aggregate tất định của B, tính lại từ artifact và không gọi model."""

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
    by_policy: dict[str, ObservationRatioSummary]
    action_counts: list[ObservationActionCountRow]
    accuracy: list[ObservationAccuracyRow]
    repeat_consistency: list[ObservationRepeatConsistencyRow]
    batch_agreement: list[ObservationBatchAgreementRow]
    candidate_relative_position: None = None
    observation_block_order: ObservationOutputs | None = None
    exclusions: list[ObservationExclusion]

    @model_validator(mode="after")
    def _summary_invariant(self) -> ObservationSummary:
        if self.artifact_schema_version != 3:
            raise ValueError("observation summaries require artifact schema version 3")
        if self.independent_question_count != 1:
            raise ValueError("this diagnostic version supports exactly one source question")
        if self.valid + self.errors + self.missing != self.scheduled:
            raise ValueError("valid, errors, and missing must partition scheduled trials")
        if self.observation_block_order is None:
            raise ValueError("observation summaries require observation output slices")
        return self


__all__ = [
    "DEFAULT_PAGE_SPLIT_NOTE",
    "DIAGNOSTIC_KIND",
    "DiagnosticExclusionReason",
    "DiagnosticKind",
    "ObservationAccuracyRow",
    "ObservationActionAgreement",
    "ObservationActionCountRow",
    "ObservationArmSpec",
    "ObservationBatchAgreementRow",
    "ObservationBatchSpec",
    "ObservationBlockSpec",
    "ObservationBlockTrace",
    "ObservationExclusion",
    "ObservationInputTrace",
    "ObservationLineOrigin",
    "ObservationManifest",
    "ObservationMovedBlock",
    "ObservationOutputs",
    "ObservationPageSplitProof",
    "ObservationPlan",
    "ObservationRatioSummary",
    "ObservationRepeatConsistencyRow",
    "ObservationSectionSpec",
    "ObservationSectionTrace",
    "ObservationSelectedBlockRow",
    "ObservationSuiteSpec",
    "ObservationSummary",
    "ObservationTransformSpec",
    "ObservationTransformTrace",
    "ObservationTrial",
]
