"""Contract riêng của sub-experiment `doctoral-defense-prompt-ablation`.

Package sở hữu registry prompt, lịch C1/C2, trace treatment và manifest C.
Bản sao arm A được mô hình hóa bằng `ReferenceArmSet` bất biến để C không phụ
thuộc code của intervention A và không cần hash nguồn lịch sử.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from diagnostic_subexp.shared.contracts import (
    ARTIFACT_ORIGIN,
    ArtifactOrigin,
    DiagnosticExecutionConfig,
    DiagnosticGateRequest,
    DiagnosticModel,
    DiagnosticResult,
    DiagnosticSourceIdentity,
    ManifestStatus,
    MessageRecord,
    NonBlank,
    NonEmptyArticleList,
    NonNegativeInt,
    PolicyName,
    PositiveInt,
    SourceReference,
    sha256_text,
    structured_hash,
)

VARIANTS = ["baseline-v1", "without-doctoral-defense-v1"]
PHASES = ["c1-q5-candidate-interaction", "c2-eligible-corpus-guard"]
ROLES = {
    "1": "stop-drift-guard",
    "2": "exclusion",
    "3": "retained-example-control",
    "4": "retained-example-control",
    "5": "targeted-probe",
    "6": "stop-drift-guard",
    "7": "general-follow-guard",
    "8": "exclusion",
}


class ReferenceArm(DiagnosticModel):
    """Bản sao bất biến một arm A được nhúng trong definition C."""

    arm_id: NonBlank
    presented_candidates: NonEmptyArticleList
    matched_pair_id: NonBlank | None = None
    relative_pair_order: str | None = None
    pair_positions: list[PositiveInt] = Field(default_factory=list)


class ReferenceBatch(DiagnosticModel):
    """Bản sao bất biến một batch A."""

    batch_id: NonBlank
    arm_order: list[NonBlank]


class ReferenceArmSet(DiagnosticModel):
    """Bản sao bất biến các identity arm/batch A mà C cần để dựng lịch."""

    suite_id: NonBlank
    source_case_identity: DiagnosticSourceIdentity
    grammar_candidates: NonEmptyArticleList
    execution_config: DiagnosticExecutionConfig
    arms: list[ReferenceArm]
    batches: list[ReferenceBatch]

    @model_validator(mode="after")
    def _arm_set_invariant(self) -> ReferenceArmSet:
        arm_ids = [arm.arm_id for arm in self.arms]
        if not self.arms or len(set(arm_ids)) != len(arm_ids):
            raise ValueError("reference arms must be nonempty with unique arm ids")
        if len(self.batches) != 2:
            raise ValueError("reference arm set requires exactly two batches")
        if self.batches[0].arm_order != arm_ids:
            raise ValueError("reference batch 0 must follow the declared arm order")
        if self.batches[1].arm_order != list(reversed(arm_ids)):
            raise ValueError("reference batch 1 must reverse the declared arm order")
        return self


class PromptAblationSpec(DiagnosticModel):
    """Definition C dựng trong bộ nhớ, không chứa hash nguồn lịch sử."""

    schema_version: Literal[1]
    suite_id: NonBlank
    diagnostic_kind: Literal["defense-fewshot-ablation"]
    case_path: NonBlank
    snapshot_path: NonBlank
    inventory_path: NonBlank
    phases: list[NonBlank]
    case_roles: dict[str, NonBlank]
    execution_config: DiagnosticExecutionConfig
    reference_arms: ReferenceArmSet
    batches: list[list[NonBlank]]
    repeats_per_input_per_batch: Literal[3]

    @model_validator(mode="after")
    def composition(self) -> PromptAblationSpec:
        if self.batches != [VARIANTS, VARIANTS[::-1]]:
            raise ValueError("C requires exactly two counterbalanced prompt variants")
        if not self.phases or self.phases != [phase for phase in PHASES if phase in self.phases]:
            raise ValueError("C phases must be an ordered nonempty subset of C1, C2")
        if self.case_roles != ROLES:
            raise ValueError("C requires the complete fixed Q1-Q8 role mapping")
        if self.execution_config != self.reference_arms.execution_config:
            raise ValueError("C and its reference A arms must share the execution config")
        return self


class PromptBaselineReference(DiagnosticGateRequest):
    """Request baseline đã capture trước refactor cùng provenance đầy đủ."""

    reference_id: NonBlank
    source_case_identity: DiagnosticSourceIdentity
    grammar_hash: NonBlank
    effective_messages_hash: NonBlank

    @model_validator(mode="after")
    def hashes(self) -> PromptBaselineReference:
        if sha256_text(self.grammar_text) != self.grammar_hash:
            raise ValueError("baseline grammar hash mismatch")
        if (
            structured_hash([message.model_dump() for message in self.effective_messages])
            != self.effective_messages_hash
        ):
            raise ValueError("baseline message hash mismatch")
        return self


class OrderedMessageIdentity(DiagnosticModel):
    """Hash và độ dài của một message theo đúng vị trí."""

    role: Literal["system", "human", "ai"]
    content_hash: NonBlank
    character_count: NonNegativeInt


class PromptRequestIdentity(DiagnosticModel):
    """Bằng chứng đầy đủ của prompt hiệu lực, không phụ thuộc renderer hiện tại."""

    variant_id: NonBlank
    included_group_ids: list[NonBlank]
    excluded_group_ids: list[NonBlank]
    ordered_messages: list[OrderedMessageIdentity]
    effective_messages: list[MessageRecord]
    effective_messages_hash: NonBlank
    message_count: NonNegativeInt
    character_count: NonNegativeInt
    system_message_hash: NonBlank
    final_human_message_hash: NonBlank


class PromptAblationInputTrace(DiagnosticModel):
    """Input bất biến độc lập biến thể prompt, dùng chung giữa các phase."""

    input_trace_id: NonBlank
    reference_id: NonBlank
    case_id: NonBlank
    source_case_identity: DiagnosticSourceIdentity
    question: str
    question_hash: NonBlank
    observation: str
    observation_hash: NonBlank
    presented_candidates: NonEmptyArticleList
    grammar_candidates: NonEmptyArticleList
    grammar_text: str
    grammar_hash: NonBlank
    execution_config_hash: NonBlank


class PromptTrace(DiagnosticModel):
    """Prompt treatment gắn với một input và complete request fingerprint."""

    prompt_trace_id: NonBlank
    input_trace_id: NonBlank
    prompt_variant_id: NonBlank
    effective_messages: list[MessageRecord]
    effective_messages_hash: NonBlank
    grammar_candidates: NonEmptyArticleList
    grammar_text: str
    grammar_hash: NonBlank
    request_fingerprint: NonBlank
    prompt_identity: PromptRequestIdentity
    baseline_parity: bool | None


class PromptAblationResult(DiagnosticResult):
    """Outcome gắn trực tiếp cả input trace và prompt trace."""

    prompt_trace_id: NonBlank


class PromptAblationTrial(DiagnosticModel):
    """Một slot C có phase, arm và treatment-pair identity riêng."""

    trial_id: NonBlank
    input_trace_id: NonBlank
    prompt_trace_id: NonBlank
    case_id: NonBlank
    diagnostic_kind: Literal["defense-fewshot-ablation"] = "defense-fewshot-ablation"
    phase_id: NonBlank
    arm_id: NonBlank
    batch_id: NonBlank
    repeat_id: NonNegativeInt
    prompt_variant_id: NonBlank
    prompt_treatment_pair_id: NonBlank
    effective_messages_hash: NonBlank
    included_example_group_ids: list[NonBlank]
    excluded_example_group_ids: list[NonBlank]
    source_a_arm_id: NonBlank | None
    case_role: NonBlank
    presented_candidates: NonEmptyArticleList
    grammar_candidates: NonEmptyArticleList
    observation_hash: NonBlank
    grammar_hash: NonBlank
    grammar_text: str
    request_fingerprint: NonBlank


class PromptExclusion(DiagnosticModel):
    """Case không có frontier chỉ dùng để accounting, không phải STOP."""

    case_id: NonBlank
    reason: Literal["no_candidates"]


class PromptAblationManifest(DiagnosticModel):
    """Manifest schema-v3 của một fresh run C."""

    artifact_origin: ArtifactOrigin
    artifact_schema_version: int = 3
    experiment: Literal["fixed-diagnostic"] = "fixed-diagnostic"
    diagnostic_kind: Literal["defense-fewshot-ablation"] = "defense-fewshot-ablation"
    run_id: NonBlank
    started_at: NonBlank
    ended_at: str | None = None
    status: ManifestStatus
    suite_id: NonBlank
    independent_question_count: PositiveInt
    policies: list[PolicyName]
    policy_config: dict[str, dict] = Field(default_factory=dict)
    execution_config: DiagnosticExecutionConfig
    sources: dict[str, SourceReference]
    execution_revision: NonBlank
    execution_dirty: bool
    executed_module_hashes: dict[str, NonBlank]
    prompt_variants: list[dict]
    phases: list[NonBlank]
    trials: list[PromptAblationTrial]
    input_trace_ids: list[NonBlank]
    prompt_trace_ids: list[NonBlank]
    exclusions: list[PromptExclusion]
    expected_trial_count: NonNegativeInt
    expected_policy_result_count: NonNegativeInt
    definition_sha256: NonBlank
    registry_sha256: NonBlank
    experiment_definition: dict
    registry_snapshot: dict

    @model_validator(mode="after")
    def counts(self) -> PromptAblationManifest:
        if self.artifact_origin != ARTIFACT_ORIGIN:
            raise ValueError(f"fresh runs require artifact_origin={ARTIFACT_ORIGIN!r}")
        if self.artifact_schema_version != 3:
            raise ValueError("fresh prompt-ablation manifests require artifact schema version 3")
        if set(self.sources) != {"case", "snapshot", "inventory"}:
            raise ValueError("fresh manifests must reference exactly case, snapshot and inventory")
        if not self.policies or len(set(self.policies)) != len(self.policies):
            raise ValueError("policies must be nonempty and unique")
        if set(self.policy_config) != set(self.policies):
            raise ValueError("policy_config keys must match policies")
        if self.expected_trial_count != len(self.trials):
            raise ValueError("trial count mismatch")
        if self.expected_policy_result_count != len(self.trials) * len(self.policies):
            raise ValueError("result count mismatch")
        if len({trial.trial_id for trial in self.trials}) != len(self.trials):
            raise ValueError("duplicate trial identity")
        if structured_hash(self.experiment_definition) != self.definition_sha256:
            raise ValueError("definition hash does not match the embedded experiment definition")
        if structured_hash(self.registry_snapshot) != self.registry_sha256:
            raise ValueError("registry hash does not match the embedded registry snapshot")
        if not self.executed_module_hashes:
            raise ValueError("fresh manifests require executed module hashes")
        for module_name, digest in self.executed_module_hashes.items():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    f"executed module hash for {module_name} is not a SHA-256 digest"
                )
        return self


__all__ = [
    "PHASES",
    "PromptAblationInputTrace",
    "PromptAblationManifest",
    "PromptAblationResult",
    "PromptAblationSpec",
    "PromptAblationTrial",
    "PromptBaselineReference",
    "PromptExclusion",
    "PromptRequestIdentity",
    "PromptTrace",
    "ROLES",
    "ReferenceArm",
    "ReferenceArmSet",
    "ReferenceBatch",
    "VARIANTS",
]
