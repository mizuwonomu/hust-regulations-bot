"""Validate definition B và dựng lịch trình Article 42 block order trong bộ nhớ.

Module chỉ xử lý intervention B: transform thứ tự observation block, kiểm tra
một control + một treatment và dựng trace bằng chứng trước/sau. Hash nguồn được
tính tại prepare từ chính bytes đọc được, không lấy từ definition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contracts import GateCase, SeedSnapshot
from seed_cases import policy_input

from diagnostic_subexp.article_42_observation_block_order.contracts import (
    DIAGNOSTIC_KIND,
    ObservationArmSpec,
    ObservationExclusion,
    ObservationInputTrace,
    ObservationPlan,
    ObservationSuiteSpec,
    ObservationTrial,
)
from diagnostic_subexp.article_42_observation_block_order.transforms import (
    transform_observation_blocks,
    verify_observation_trace,
)
from diagnostic_subexp.shared.contracts import (
    SourceReference,
    sha256_text,
    structured_hash,
)
from diagnostic_subexp.shared.execution import (
    build_effective_messages,
    build_grammar_text,
    compute_request_fingerprint,
)
from diagnostic_subexp.shared.source_refs import (
    resolve_repository_source,
    sha256_file,
)

VALID_POLICIES = ("first", "llm")
# Bằng chứng của Article 42 có ghi lại pair slot Article 3/41
PAIR_ARTICLES: tuple[int, int] = (3, 41)

_EXECUTION_CONFIG: dict[str, Any] = {
    "model_alias": "citation-agent",
    "base_url": "http://127.0.0.1:8080/v1",
    "temperature": 0.0,
    "max_completion_tokens": 4096,
    "enable_thinking": False,
    "timeout_seconds": None,
    "max_retries": None,
    "extra_request_options": {"chat_template_kwargs": {"enable_thinking": False}},
}

_SOURCE_CASE_IDENTITY: dict[str, Any] = {
    "dataset_id": "corpus_cross_references",
    "question_id": 5,
    "case_id": "corpus_cross_references:hop0:q-ef2d127de37b",
    "source_hop": 0,
}

_ARMS: list[dict[str, Any]] = [
    {
        "arm_id": "b0-original-control",
        "presented_candidates": [41, 3, 40, 8, 43],
        "observation_transform_id": "b-identity",
        "matched_pair_id": "b42-block-order",
    },
    {
        "arm_id": "b1-reversed-blocks",
        "presented_candidates": [41, 3, 40, 8, 43],
        "observation_transform_id": "b-reversed-blocks",
        "matched_pair_id": "b42-block-order",
    },
]

_OBSERVATION_BLOCKS: list[dict[str, Any]] = [
    {
        "block_id": "b42-foundation-timing",
        "context_article_id": 42,
        "source_line_count": 2,
        "source_line_numbers": [3, 4],
        "source_byte_span": [114, 659],
        "source_text_sha256": "72a8156f1ecbaa2c3f7442574389fb6c3e2c957c03007db00ad40bbc51346c1a",
        "referenced_candidates": [41, 3],
        "must_remain_contiguous": True,
    },
    {
        "block_id": "b42-university-defense",
        "context_article_id": 42,
        "source_line_count": 1,
        "source_line_numbers": [5],
        "source_byte_span": [659, 946],
        "source_text_sha256": "7dea2278ac343e0ac906c8aa6cc34ce31e382ac313d9fc2a67492834237bf5ab",
        "referenced_candidates": [40],
        "must_remain_contiguous": True,
    },
]

_OBSERVATION_SECTIONS: list[dict[str, Any]] = [
    {
        "section_id": "obs-preamble",
        "source_line_numbers": [1, 2],
        "source_byte_span": [0, 114],
        "source_text_sha256": "822f578bfa2df76d0fdf90d9ea3d604f7d24980be23994373dbb97a96d952db2",
        "referenced_candidates": [],
    },
    {
        "section_id": "ctx45-section",
        "source_line_numbers": [6, 7, 8],
        "source_byte_span": [946, 1427],
        "source_text_sha256": "d7958e277a3f1eca3bdd8b79bec3ff763ed58b6c992571d04ee51b0087f65ffd",
        "referenced_candidates": [8, 43],
    },
]

_PAGE_SPLIT_GROUP: dict[str, Any] = {
    "group_id": "b42-foundation-timing-page-split",
    "block_id": "b42-foundation-timing",
    "source_line_numbers": [3, 4],
    "line_sha256": [
        "ecf9c573736048571de62e435bec736eed9ac4233e553b55c49385b4f19ec88a",
        "fa30ddecb4b25dd46829532022274693909a930c8fe9a1dc537f12647d3e005b",
    ],
}

_OBSERVATION_TRANSFORMS: list[dict[str, Any]] = [
    {
        "transform_id": "b-identity",
        "ordered_block_ids": ["b42-foundation-timing", "b42-university-defense"],
        "prefix_byte_span": [0, 114],
        "prefix_sha256": "822f578bfa2df76d0fdf90d9ea3d604f7d24980be23994373dbb97a96d952db2",
        "suffix_byte_span": [946, 1427],
        "suffix_sha256": "d7958e277a3f1eca3bdd8b79bec3ff763ed58b6c992571d04ee51b0087f65ffd",
    },
    {
        "transform_id": "b-reversed-blocks",
        "ordered_block_ids": ["b42-university-defense", "b42-foundation-timing"],
        "prefix_byte_span": [0, 114],
        "prefix_sha256": "822f578bfa2df76d0fdf90d9ea3d604f7d24980be23994373dbb97a96d952db2",
        "suffix_byte_span": [946, 1427],
        "suffix_sha256": "d7958e277a3f1eca3bdd8b79bec3ff763ed58b6c992571d04ee51b0087f65ffd",
    },
]

_BATCHES: list[dict[str, Any]] = [
    {"batch_id": "batch-0", "arm_order": [arm["arm_id"] for arm in _ARMS]},
    {"batch_id": "batch-1", "arm_order": [arm["arm_id"] for arm in reversed(_ARMS)]},
]


def derive_pair_metadata(presented: list[int]) -> tuple[list[int], str]:
    """Suy ra pair slots 1-based và orientation 3/41 từ presented order."""
    positions = sorted(
        presented.index(article) + 1 for article in PAIR_ARTICLES if article in presented
    )
    if len(positions) != 2:
        raise ValueError(f"presented candidates must contain both articles {PAIR_ARTICLES}")
    orientation = "3-41" if presented.index(3) < presented.index(41) else "41-3"
    return positions, orientation


def build_observation_definition() -> dict[str, Any]:
    """Dựng definition B hoàn chỉnh trong bộ nhớ, không chứa hash nguồn lịch sử."""
    return {
        "schema_version": 1,
        "suite_id": "fixed-diagnostic-b-v1",
        "diagnostic_kind": DIAGNOSTIC_KIND,
        "case_path": "evals/agent-exp/datasets/gate_cases_fixed_v1.jsonl",
        "snapshot_path": "evals/agent-exp/snapshots/seeds_v1_fixed.json",
        "inventory_path": "evals/agent-exp/datasets/internal_dieu_fixed.json",
        "source_case_identity": json.loads(json.dumps(_SOURCE_CASE_IDENTITY)),
        "execution_config": json.loads(json.dumps(_EXECUTION_CONFIG)),
        "grammar_candidates": list(_ARMS[0]["presented_candidates"]),
        "arms": json.loads(json.dumps(_ARMS)),
        "observation_blocks": json.loads(json.dumps(_OBSERVATION_BLOCKS)),
        "observation_sections": json.loads(json.dumps(_OBSERVATION_SECTIONS)),
        "page_split_group": json.loads(json.dumps(_PAGE_SPLIT_GROUP)),
        "observation_transforms": json.loads(json.dumps(_OBSERVATION_TRANSFORMS)),
        "batches": json.loads(json.dumps(_BATCHES)),
        "repeats_per_input_per_batch": 3,
    }


def read_observation_spec(path: Path | str | None = None) -> ObservationSuiteSpec:
    """Đọc definition B nhúng trong bộ nhớ hoặc một file JSON tương thích."""
    if path is None:
        return ObservationSuiteSpec.model_validate(build_observation_definition())
    path = Path(path)
    if not path.is_file():
        return ObservationSuiteSpec.model_validate(build_observation_definition())
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON at line {exc.lineno}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: the diagnostic specification must be a JSON object")
    version = payload.get("schema_version")
    if version != 1:
        raise ValueError(
            f"{path}: unsupported diagnostic schema version: {version!r}, "
            f"supported version is 1"
        )
    try:
        return ObservationSuiteSpec.model_validate(payload)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid diagnostic specification: {exc}") from exc


def question_identity(value: Any) -> tuple[type[Any], Any]:
    """Giữ nguyên kiểu của question id khi so khớp nguồn."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"question id must be an integer or string: {value!r}")
    return type(value), value


def resolve_observation_sources(spec: ObservationSuiteSpec) -> dict[str, Path]:
    """Resolve ba nguồn nền canonical từ repository root."""
    return {
        "case": resolve_repository_source(spec.case_path),
        "snapshot": resolve_repository_source(spec.snapshot_path),
        "inventory": resolve_repository_source(spec.inventory_path),
    }


def validate_observation_arm_schedule(
    arms: list[ObservationArmSpec],
    spec: ObservationSuiteSpec,
) -> None:
    """Kiểm tra B có một control, một treatment và đúng hai transform."""
    pairs: dict[str, list[ObservationArmSpec]] = {}
    for arm in arms:
        if arm.matched_pair_id is None:
            continue
        pairs.setdefault(arm.matched_pair_id, []).append(arm)
    if len(pairs) != 1:
        raise ValueError("intervention B requires exactly one matched arm pair")
    members = next(iter(pairs.values()))
    if len(members) != 2:
        raise ValueError("intervention B requires exactly two arms in its matched pair")
    before_order = [block.block_id for block in spec.observation_blocks]
    identity_members = [
        arm for arm in members if _transform_order(spec, arm.observation_transform_id) == before_order
    ]
    if len(identity_members) != 1:
        raise ValueError("intervention B requires exactly one identity-order control arm")
    order_a = _transform_order(spec, members[0].observation_transform_id)
    order_b = _transform_order(spec, members[1].observation_transform_id)
    if order_a == order_b:
        raise ValueError("intervention B requires the two arms to use different block orders")


def _transform_order(spec: ObservationSuiteSpec, transform_id: str) -> list[str]:
    for transform in spec.observation_transforms:
        if transform.transform_id == transform_id:
            return list(transform.ordered_block_ids)
    raise ValueError(f"unknown observation transform: {transform_id!r}")


def _select_source_case(spec: ObservationSuiteSpec, cases: list[GateCase]) -> GateCase:
    identity = spec.source_case_identity
    matches = [case for case in cases if case.case_id == identity.case_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one source case {identity.case_id!r}, found {len(matches)}"
        )
    case = matches[0]
    if case.dataset_id != identity.dataset_id:
        raise ValueError("source case dataset id does not match the specification")
    if question_identity(case.question_id) != question_identity(identity.question_id):
        raise ValueError("source case question id does not match the specification")
    if case.source_hop != identity.source_hop:
        raise ValueError("source case hop does not match the specification")
    return case


def _validate_case_labels(case: GateCase) -> None:
    if case.label_status != "approved":
        raise ValueError(f"{case.case_id}: diagnostic scoring requires an approved label")
    if case.expected_action not in {"follow", "stop"}:
        raise ValueError(f"{case.case_id}: unsupported semantic label {case.expected_action!r}")
    if sha256_text(case.observation) != case.observation_hash:
        raise ValueError(f"{case.case_id}: stored observation hash does not match its bytes")


def _source_references(spec: ObservationSuiteSpec, sources: dict[str, Path]) -> dict[str, SourceReference]:
    return {
        label: SourceReference(
            role=label,
            path=spec_path,
            sha256=sha256_file(path),
        )
        for label, (spec_path, path) in {
            "case": (spec.case_path, sources["case"]),
            "snapshot": (spec.snapshot_path, sources["snapshot"]),
            "inventory": (spec.inventory_path, sources["inventory"]),
        }.items()
    }


def build_observation_trace(
    case: GateCase,
    arm: ObservationArmSpec,
    spec: ObservationSuiteSpec,
    *,
    execution_config_hash: str,
) -> ObservationInputTrace:
    """Dựng trace bền vững cho một arm B, không đọc nhãn."""
    source_input = policy_input(case)
    observation_trace = transform_observation_blocks(source_input, arm, spec)
    observation = observation_trace.transformed_observation
    pair_positions, relative_pair_order = derive_pair_metadata(list(arm.presented_candidates))
    messages = build_effective_messages(
        source_input.question, observation, list(arm.presented_candidates)
    )
    grammar_text = build_grammar_text(spec.grammar_candidates)
    fingerprint = compute_request_fingerprint(
        spec.source_case_identity,
        messages,
        list(arm.presented_candidates),
        grammar_text,
        spec.execution_config,
    )
    return ObservationInputTrace(
        input_trace_id=f"{case.case_id}:{DIAGNOSTIC_KIND}:{arm.arm_id}",
        case_id=case.case_id,
        source_case_identity=spec.source_case_identity,
        diagnostic_kind=DIAGNOSTIC_KIND,
        arm_id=arm.arm_id,
        matched_pair_id=arm.matched_pair_id,
        pair_positions=list(pair_positions),
        relative_pair_order=relative_pair_order,
        question_hash=sha256_text(source_input.question),
        original_candidates=list(source_input.candidates),
        presented_candidates=list(arm.presented_candidates),
        grammar_candidates=list(spec.grammar_candidates),
        original_observation_hash=sha256_text(observation_trace.original_observation),
        transformed_observation_hash=sha256_text(observation),
        effective_messages=messages,
        grammar_text=grammar_text,
        effective_messages_hash=structured_hash(
            [message.model_dump(mode="json") for message in messages]
        ),
        grammar_hash=sha256_text(grammar_text),
        request_fingerprint=fingerprint,
        execution_config_hash=execution_config_hash,
        declared_changed_fields=list(observation_trace.declared_changed_fields),
        verified_unchanged_fields=list(observation_trace.verified_unchanged_fields),
        original_observation=observation_trace.original_observation,
        transformed_observation=observation_trace.transformed_observation,
        before_block_order=list(observation_trace.before_block_order),
        after_block_order=list(observation_trace.after_block_order),
        blocks=list(observation_trace.blocks),
        moved_blocks=list(observation_trace.moved_blocks),
        line_origin_map=list(observation_trace.line_origin_map),
        source_and_output_byte_spans={
            block.block_id: {
                "before": list(block.before_byte_span),
                "after": list(block.after_byte_span),
            }
            for block in observation_trace.blocks
        },
        unchanged_context_sections=list(observation_trace.sections),
        page_split_group=observation_trace.page_split_group,
    )


def plan_observation_run(
    spec: ObservationSuiteSpec,
    cases: list[GateCase],
    snapshot: SeedSnapshot,
    policies: list[str],
) -> ObservationPlan:
    """Kiểm tra provenance rồi lập lịch tất định cho intervention B."""
    if not isinstance(spec, ObservationSuiteSpec):
        raise TypeError("spec must be an ObservationSuiteSpec")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list of GateCase")
    if not isinstance(snapshot, SeedSnapshot):
        raise TypeError("snapshot must be a SeedSnapshot")
    policy_list = list(policies)
    if not policy_list:
        raise ValueError("at least one policy is required")
    if len(set(policy_list)) != len(policy_list):
        raise ValueError("policies must not contain duplicates")
    for policy in policy_list:
        if policy not in VALID_POLICIES:
            raise ValueError(f"unsupported policy: {policy!r}")

    sources = resolve_observation_sources(spec)
    source_references = _source_references(spec, sources)
    if snapshot.dataset_id != spec.source_case_identity.dataset_id:
        raise ValueError("snapshot dataset id does not match the source case identity")

    case = _select_source_case(spec, cases)
    _validate_case_labels(case)
    if case.snapshot_id != snapshot.snapshot_id:
        raise ValueError("source case snapshot id does not match the canonical snapshot")
    if case.snapshot_hash != source_references["snapshot"].sha256:
        raise ValueError("source case snapshot hash does not match the canonical snapshot")
    if list(case.candidates) != list(spec.grammar_candidates):
        raise ValueError("the source case candidate list must equal the fixed grammar candidates")
    validate_observation_arm_schedule(spec.arms, spec)

    execution_config_hash = structured_hash(spec.execution_config.model_dump(mode="json"))
    traces = [
        build_observation_trace(case, arm, spec, execution_config_hash=execution_config_hash)
        for arm in spec.arms
    ]
    trace_by_arm = {trace.arm_id: trace for trace in traces}

    trials: list[ObservationTrial] = []
    for batch in spec.batches:
        for repeat_id in range(spec.repeats_per_input_per_batch):
            for arm_id in batch.arm_order:
                trace = trace_by_arm[arm_id]
                trials.append(
                    ObservationTrial(
                        trial_id=(
                            f"{case.case_id}:{DIAGNOSTIC_KIND}:{arm_id}:"
                            f"{batch.batch_id}:r{repeat_id}"
                        ),
                        input_trace_id=trace.input_trace_id,
                        case_id=case.case_id,
                        diagnostic_kind=DIAGNOSTIC_KIND,
                        arm_id=arm_id,
                        batch_id=batch.batch_id,
                        repeat_id=repeat_id,
                        matched_pair_id=trace.matched_pair_id,
                        pair_positions=list(trace.pair_positions),
                        relative_pair_order=trace.relative_pair_order,
                        presented_candidates=list(trace.presented_candidates),
                        grammar_candidates=list(trace.grammar_candidates),
                        observation_hash=trace.transformed_observation_hash,
                        effective_messages_hash=trace.effective_messages_hash,
                        grammar_hash=trace.grammar_hash,
                        request_fingerprint=trace.request_fingerprint,
                    )
                )

    scheduled_configurations = {tuple(trial.presented_candidates) for trial in trials}
    declared_configurations = {tuple(arm.presented_candidates) for arm in spec.arms}
    if scheduled_configurations != declared_configurations:
        raise ValueError("batch and repeat expansion created a new unique candidate configuration")

    exclusions = [
        ObservationExclusion(case_id=other.case_id, reason="not_selected_case")
        for other in cases
        if other.case_id != case.case_id
    ]
    return ObservationPlan(
        suite_id=spec.suite_id,
        diagnostic_kind=DIAGNOSTIC_KIND,
        source_case_identity=spec.source_case_identity,
        case_source=source_references["case"],
        snapshot_source=source_references["snapshot"],
        inventory_source=source_references["inventory"],
        policies=policy_list,
        execution_config=spec.execution_config,
        traces=traces,
        trials=trials,
        exclusions=exclusions,
        expected_trial_count=len(trials),
        expected_policy_result_count=len(trials) * len(policy_list),
    )


def verify_observation_input_trace(
    trace: ObservationInputTrace,
    case: GateCase,
    arm: ObservationArmSpec,
    spec: ObservationSuiteSpec,
) -> None:
    """Dựng lại transform observation và đối chiếu trace đã lưu."""
    source_input = policy_input(case)
    if trace.case_id != case.case_id:
        raise ValueError("trace case id does not match the bundled case")
    if trace.arm_id != arm.arm_id:
        raise ValueError("trace arm id does not match the bundled specification")
    if trace.diagnostic_kind != spec.diagnostic_kind:
        raise ValueError("trace diagnostic kind does not match the specification")
    if trace.source_case_identity != spec.source_case_identity:
        raise ValueError("trace source identity does not match the specification")
    if trace.question_hash != sha256_text(source_input.question):
        raise ValueError("trace question hash does not match the bundled case")
    if trace.original_observation_hash != sha256_text(source_input.observation):
        raise ValueError("trace original observation hash does not match the bundled case")
    if trace.grammar_candidates != list(spec.grammar_candidates):
        raise ValueError("trace grammar candidates do not match the specification")
    if trace.execution_config_hash != structured_hash(spec.execution_config.model_dump(mode="json")):
        raise ValueError("trace execution config hash does not match the specification")
    if trace.original_candidates != list(source_input.candidates):
        raise ValueError("trace original candidates do not match the bundled case")
    if trace.presented_candidates != list(arm.presented_candidates):
        raise ValueError("trace presented candidates do not match the specification")
    pair_positions, relative_pair_order = derive_pair_metadata(list(trace.presented_candidates))
    if trace.pair_positions != pair_positions or trace.relative_pair_order != relative_pair_order:
        raise ValueError("trace pair metadata does not match the presented order")

    verify_observation_trace(trace, source_input, arm, spec)

    expected_fingerprint = compute_request_fingerprint(
        trace.source_case_identity,
        trace.effective_messages,
        trace.presented_candidates,
        trace.grammar_text,
        spec.execution_config,
    )
    if trace.request_fingerprint != expected_fingerprint:
        raise ValueError("trace request fingerprint does not match the stored request")


__all__ = [
    "VALID_POLICIES",
    "build_observation_definition",
    "build_observation_trace",
    "compute_request_fingerprint",
    "derive_pair_metadata",
    "plan_observation_run",
    "question_identity",
    "read_observation_spec",
    "resolve_observation_sources",
    "validate_observation_arm_schedule",
    "verify_observation_input_trace",
]
