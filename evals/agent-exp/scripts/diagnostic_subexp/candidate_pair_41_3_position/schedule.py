"""Validate definition A và dựng lịch trình candidate-pair 41/3 trong bộ nhớ.

Module chỉ xử lý intervention A: transform thứ tự candidate, kiểm tra strata
cặp 3/41, dựng input trace và lịch chạy tất định từ các nguồn canonical dùng
chung. Hash nguồn được tính tại prepare từ chính bytes đọc được, không lấy từ
definition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contracts import GateCase, SeedSnapshot
from seed_cases import policy_input

from diagnostic_subexp.candidate_pair_41_3_position.contracts import (
    CandidateArmSpec,
    CandidateExclusion,
    CandidateInputTrace,
    CandidatePlan,
    CandidateSuiteSpec,
    CandidateTrial,
    DIAGNOSTIC_KIND,
    PAIR_ARTICLES,
    derive_pair_metadata,
)
from diagnostic_subexp.candidate_pair_41_3_position.transforms import (
    transform_candidate_order,
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
        "arm_id": "a0-original-control",
        "presented_candidates": [41, 3, 40, 8, 43],
        "matched_pair_id": None,
        "relative_pair_order": None,
        "pair_positions": [],
    },
    {
        "arm_id": "a1-pair-23-3-first",
        "presented_candidates": [8, 3, 41, 40, 43],
        "matched_pair_id": "a-pair-23",
        "relative_pair_order": "3-41",
        "pair_positions": [2, 3],
    },
    {
        "arm_id": "a2-pair-23-41-first",
        "presented_candidates": [8, 41, 3, 40, 43],
        "matched_pair_id": "a-pair-23",
        "relative_pair_order": "41-3",
        "pair_positions": [2, 3],
    },
    {
        "arm_id": "a3-pair-34-3-first",
        "presented_candidates": [8, 40, 3, 41, 43],
        "matched_pair_id": "a-pair-34",
        "relative_pair_order": "3-41",
        "pair_positions": [3, 4],
    },
    {
        "arm_id": "a4-pair-34-41-first",
        "presented_candidates": [8, 40, 41, 3, 43],
        "matched_pair_id": "a-pair-34",
        "relative_pair_order": "41-3",
        "pair_positions": [3, 4],
    },
]

_BATCHES: list[dict[str, Any]] = [
    {
        "batch_id": "batch-0",
        "arm_order": [arm["arm_id"] for arm in _ARMS],
    },
    {
        "batch_id": "batch-1",
        "arm_order": [arm["arm_id"] for arm in reversed(_ARMS)],
    },
]


def build_candidate_definition() -> dict[str, Any]:
    """Dựng definition A hoàn chỉnh trong bộ nhớ, không chứa hash nguồn lịch sử."""
    return {
        "schema_version": 1,
        "suite_id": "fixed-diagnostic-a-v1",
        "diagnostic_kind": DIAGNOSTIC_KIND,
        "case_path": "evals/agent-exp/datasets/gate_cases_fixed_v1.jsonl",
        "snapshot_path": "evals/agent-exp/snapshots/seeds_v1_fixed.json",
        "inventory_path": "evals/agent-exp/datasets/internal_dieu_fixed.json",
        "source_case_identity": json.loads(json.dumps(_SOURCE_CASE_IDENTITY)),
        "execution_config": json.loads(json.dumps(_EXECUTION_CONFIG)),
        "grammar_candidates": list(_ARMS[0]["presented_candidates"]),
        "arms": json.loads(json.dumps(_ARMS)),
        "batches": json.loads(json.dumps(_BATCHES)),
        "repeats_per_input_per_batch": 3,
    }


def read_candidate_spec(path: Path | str | None = None) -> CandidateSuiteSpec:
    """Đọc definition A nhúng trong bộ nhớ hoặc một file JSON tương thích."""
    if path is None:
        return CandidateSuiteSpec.model_validate(build_candidate_definition())
    path = Path(path)
    if not path.is_file():
        return CandidateSuiteSpec.model_validate(build_candidate_definition())
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
        return CandidateSuiteSpec.model_validate(payload)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid diagnostic specification: {exc}") from exc


def question_identity(value: Any) -> tuple[type[Any], Any]:
    """Giữ nguyên kiểu của question id khi so khớp nguồn."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"question id must be an integer or string: {value!r}")
    return type(value), value


def resolve_candidate_sources(spec: CandidateSuiteSpec) -> dict[str, Path]:
    """Resolve ba nguồn nền canonical từ repository root."""
    return {
        "case": resolve_repository_source(spec.case_path),
        "snapshot": resolve_repository_source(spec.snapshot_path),
        "inventory": resolve_repository_source(spec.inventory_path),
    }


def validate_candidate_arm_schedule(arms: list[CandidateArmSpec]) -> list[dict[str, Any]]:
    """Kiểm tra lịch A: hai slot strata, orientation cân bằng trong từng strata."""
    controls = [arm for arm in arms if arm.matched_pair_id is None]
    if len(controls) != 1:
        raise ValueError("intervention A requires exactly one original control arm")
    treatments = [arm for arm in arms if arm.matched_pair_id is not None]
    if len(treatments) < 2:
        raise ValueError("intervention A requires matched treatment arms")
    if len(treatments) % 2 != 0:
        raise ValueError("intervention A requires an even number of treatment arms")

    strata: dict[tuple[int, int], list[CandidateArmSpec]] = {}
    for arm in treatments:
        positions = tuple(arm.pair_positions)
        if len(positions) != 2:
            raise ValueError(f"arm {arm.arm_id} must declare exactly two pair positions")
        if positions[1] - positions[0] != 1:
            raise ValueError(f"arm {arm.arm_id} must keep the Article 3/41 pair adjacent")
        strata.setdefault(positions, []).append(arm)

    if len(strata) != 2:
        raise ValueError("intervention A requires exactly two matched slot strata")
    for positions, members in sorted(strata.items()):
        if len(members) != 2:
            raise ValueError(f"slot stratum {positions} must contain exactly two treatment arms")
        orientations = sorted(
            member.relative_pair_order for member in members if member.relative_pair_order is not None
        )
        if orientations != ["3-41", "41-3"]:
            raise ValueError(
                f"slot stratum {positions} must contain one orientation of each pair order"
            )
        reference: dict[int, int] | None = None
        for member in members:
            others = {
                position: candidate
                for position, candidate in enumerate(member.presented_candidates, start=1)
                if candidate not in PAIR_ARTICLES
            }
            if reference is None:
                reference = others
            elif others != reference:
                raise ValueError(
                    f"slot stratum {positions} moved a non-pair candidate inside a matched swap"
                )

    orders = [tuple(arm.presented_candidates) for arm in treatments]
    if len(set(orders)) != len(orders):
        raise ValueError("intervention A requires unique presented candidate orders")
    if tuple(controls[0].presented_candidates) in orders:
        raise ValueError("the original control must not duplicate a treatment order")

    return [
        {
            "pair_positions": list(positions),
            "pair_ids": sorted({member.matched_pair_id for member in members}),
            "arms": [member.arm_id for member in members],
            "orientations": sorted(member.relative_pair_order for member in members),
        }
        for positions, members in sorted(strata.items())
    ]


def _select_source_case(spec: CandidateSuiteSpec, cases: list[GateCase]) -> GateCase:
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


def _source_references(spec: CandidateSuiteSpec, sources: dict[str, Path]) -> dict[str, SourceReference]:
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


def build_candidate_trace(
    case: GateCase,
    arm: CandidateArmSpec,
    spec: CandidateSuiteSpec,
    *,
    execution_config_hash: str,
) -> CandidateInputTrace:
    """Dựng trace bền vững cho một arm A, không đọc nhãn."""
    source_input = policy_input(case)
    candidate_trace = transform_candidate_order(source_input, arm)
    observation = source_input.observation
    messages = build_effective_messages(
        source_input.question,
        observation,
        candidate_trace.presented_candidates,
    )
    grammar_text = build_grammar_text(spec.grammar_candidates)
    fingerprint = compute_request_fingerprint(
        spec.source_case_identity,
        messages,
        candidate_trace.presented_candidates,
        grammar_text,
        spec.execution_config,
    )
    return CandidateInputTrace(
        input_trace_id=f"{case.case_id}:{DIAGNOSTIC_KIND}:{arm.arm_id}",
        case_id=case.case_id,
        source_case_identity=spec.source_case_identity,
        diagnostic_kind=DIAGNOSTIC_KIND,
        arm_id=arm.arm_id,
        matched_pair_id=arm.matched_pair_id,
        pair_positions=list(candidate_trace.pair_positions),
        relative_pair_order=candidate_trace.relative_pair_order,
        question_hash=sha256_text(source_input.question),
        original_candidates=list(candidate_trace.original_candidates),
        presented_candidates=list(candidate_trace.presented_candidates),
        grammar_candidates=list(spec.grammar_candidates),
        original_observation_hash=candidate_trace.original_observation_hash,
        transformed_observation_hash=sha256_text(observation),
        effective_messages=messages,
        grammar_text=grammar_text,
        effective_messages_hash=structured_hash(
            [message.model_dump(mode="json") for message in messages]
        ),
        grammar_hash=sha256_text(grammar_text),
        request_fingerprint=fingerprint,
        execution_config_hash=execution_config_hash,
        declared_changed_fields=list(candidate_trace.declared_changed_fields),
        verified_unchanged_fields=list(candidate_trace.verified_unchanged_fields),
    )


def plan_candidate_run(
    spec: CandidateSuiteSpec,
    cases: list[GateCase],
    snapshot: SeedSnapshot,
    policies: list[str],
) -> CandidatePlan:
    """Kiểm tra provenance rồi lập lịch tất định cho intervention A."""
    if not isinstance(spec, CandidateSuiteSpec):
        raise TypeError("spec must be a CandidateSuiteSpec")
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

    sources = resolve_candidate_sources(spec)
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
    validate_candidate_arm_schedule(spec.arms)

    execution_config_hash = structured_hash(spec.execution_config.model_dump(mode="json"))
    traces = [
        build_candidate_trace(case, arm, spec, execution_config_hash=execution_config_hash)
        for arm in spec.arms
    ]
    trace_by_arm = {trace.arm_id: trace for trace in traces}

    trials: list[CandidateTrial] = []
    for batch in spec.batches:
        for repeat_id in range(spec.repeats_per_input_per_batch):
            for arm_id in batch.arm_order:
                trace = trace_by_arm[arm_id]
                trials.append(
                    CandidateTrial(
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
        CandidateExclusion(case_id=other.case_id, reason="not_selected_case")
        for other in cases
        if other.case_id != case.case_id
    ]
    return CandidatePlan(
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


def verify_candidate_trace(
    trace: CandidateInputTrace,
    case: GateCase,
    arm: CandidateArmSpec,
    spec: CandidateSuiteSpec,
) -> None:
    """Dựng lại transform candidate và đối chiếu trace đã lưu."""
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

    candidate_trace = transform_candidate_order(source_input, arm)
    if trace.original_candidates != list(candidate_trace.original_candidates):
        raise ValueError("trace original candidates do not match the reconstruction")
    if trace.presented_candidates != list(candidate_trace.presented_candidates):
        raise ValueError("trace presented candidates do not match the reconstruction")
    if trace.pair_positions != list(candidate_trace.pair_positions):
        raise ValueError("trace pair_positions do not match the reconstruction")
    if trace.relative_pair_order != candidate_trace.relative_pair_order:
        raise ValueError("trace relative_pair_order does not match the reconstruction")
    if trace.declared_changed_fields != list(candidate_trace.declared_changed_fields):
        raise ValueError("trace declared changed fields do not match the reconstruction")
    if trace.verified_unchanged_fields != list(candidate_trace.verified_unchanged_fields):
        raise ValueError("trace verified unchanged fields do not match the reconstruction")
    if trace.transformed_observation_hash != sha256_text(source_input.observation):
        raise ValueError("trace transformed observation hash does not match the reconstruction")

    expected_fingerprint = compute_request_fingerprint(
        trace.source_case_identity,
        trace.effective_messages,
        trace.presented_candidates,
        trace.grammar_text,
        spec.execution_config,
    )
    if trace.request_fingerprint != expected_fingerprint:
        raise ValueError("trace request fingerprint does not match the stored request")

    derived_positions, derived_order = derive_pair_metadata(list(trace.presented_candidates))
    if trace.pair_positions != derived_positions or trace.relative_pair_order != derived_order:
        raise ValueError("trace pair metadata does not match the presented order")


__all__ = [
    "VALID_POLICIES",
    "build_candidate_definition",
    "build_candidate_trace",
    "compute_request_fingerprint",
    "plan_candidate_run",
    "question_identity",
    "read_candidate_spec",
    "resolve_candidate_sources",
    "validate_candidate_arm_schedule",
    "verify_candidate_trace",
]
