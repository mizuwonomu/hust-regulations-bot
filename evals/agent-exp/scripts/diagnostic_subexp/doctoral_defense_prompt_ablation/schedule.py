"""Dựng và kiểm tra lịch C bằng nguồn cố định, không đọc model results.

Module dựng definition C trong bộ nhớ từ registry và baseline fixture do eval
sở hữu; các arm A được sao chép như giá trị bất biến nên C không phụ thuộc code
của intervention A.
"""

from __future__ import annotations

import json
from typing import Any

from diagnostic_subexp.doctoral_defense_prompt_ablation.contracts import (
    PHASES,
    ROLES,
    VARIANTS,
    PromptAblationInputTrace,
    PromptAblationSpec,
    PromptAblationTrial,
    PromptRequestIdentity,
    PromptTrace,
    ReferenceArmSet,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.variants import (
    baseline_references,
    prompt_registry,
)
from diagnostic_subexp.shared.contracts import (
    DiagnosticSourceIdentity,
    MessageRecord,
    sha256_text,
    structured_hash,
)
from diagnostic_subexp.shared.execution import compute_request_fingerprint

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

_REFERENCE_ARMS: list[dict[str, Any]] = [
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

_REFERENCE_BATCHES: list[dict[str, Any]] = [
    {"batch_id": "batch-0", "arm_order": [arm["arm_id"] for arm in _REFERENCE_ARMS]},
    {"batch_id": "batch-1", "arm_order": [arm["arm_id"] for arm in reversed(_REFERENCE_ARMS)]},
]


def build_prompt_definition() -> dict[str, Any]:
    """Dựng definition C hoàn chỉnh trong bộ nhớ, không chứa hash nguồn lịch sử."""
    reference_arms = {
        "suite_id": "fixed-diagnostic-a-v1",
        "source_case_identity": {
            "dataset_id": "corpus_cross_references",
            "question_id": 5,
            "case_id": "corpus_cross_references:hop0:q-ef2d127de37b",
            "source_hop": 0,
        },
        "grammar_candidates": list(_REFERENCE_ARMS[0]["presented_candidates"]),
        "execution_config": json.loads(json.dumps(_EXECUTION_CONFIG)),
        "arms": json.loads(json.dumps(_REFERENCE_ARMS)),
        "batches": json.loads(json.dumps(_REFERENCE_BATCHES)),
    }
    return {
        "schema_version": 1,
        "suite_id": "fixed-diagnostic-c-v1",
        "diagnostic_kind": "defense-fewshot-ablation",
        "case_path": "evals/agent-exp/datasets/gate_cases_fixed_v1.jsonl",
        "snapshot_path": "evals/agent-exp/snapshots/seeds_v1_fixed.json",
        "inventory_path": "evals/agent-exp/datasets/internal_dieu_fixed.json",
        "phases": list(PHASES),
        "case_roles": dict(ROLES),
        "execution_config": json.loads(json.dumps(_EXECUTION_CONFIG)),
        "reference_arms": reference_arms,
        "batches": [list(VARIANTS), list(reversed(VARIANTS))],
        "repeats_per_input_per_batch": 3,
        "prompt_variants": prompt_registry()["variants"],
        "registry_snapshot": prompt_registry(),
        "baseline_references": baseline_references(),
    }


def source_identity(case: Any) -> DiagnosticSourceIdentity:
    """Lấy identity đầy đủ và giữ nguyên kiểu question id."""
    return DiagnosticSourceIdentity(
        **{
            key: getattr(case, key)
            for key in ("dataset_id", "question_id", "case_id", "source_hop")
        }
    )


def _registry_messages(registry: dict[str, Any], group_ids: list[str]) -> list[dict[str, str]]:
    """Dựng literal message có thứ tự cho các group ví dụ được chọn."""
    by_id = {group["group_id"]: group for group in registry["groups"]}
    rows: list[dict[str, str]] = []
    for group_id in group_ids:
        for pair in by_id[group_id]["pairs"]:
            rows.extend(
                [
                    {"role": "human", "content": pair["human_content"]},
                    {"role": "ai", "content": pair["assistant_content"]},
                ]
            )
    return rows


def prompt_request_identity(
    messages: list[MessageRecord],
    variant: dict[str, Any],
) -> PromptRequestIdentity:
    """Tính identity từ message literal và composition của variant."""
    rows = [message.model_dump(mode="json") for message in messages]
    return PromptRequestIdentity(
        variant_id=variant["variant_id"],
        included_group_ids=variant["included_group_ids"],
        excluded_group_ids=variant["excluded_group_ids"],
        effective_messages=messages,
        ordered_messages=[
            {
                "role": message.role,
                "content_hash": sha256_text(message.content),
                "character_count": len(message.content),
            }
            for message in messages
        ],
        effective_messages_hash=structured_hash(rows),
        message_count=len(rows),
        character_count=sum(len(message.content) for message in messages),
        system_message_hash=sha256_text(messages[0].content),
        final_human_message_hash=sha256_text(messages[-1].content),
    )


def validate_registry(
    registry: dict[str, Any],
    references: list[Any],
) -> list[dict[str, Any]]:
    """Kiểm tra exact group/pair removal từ registry và baseline đã lưu."""
    from src.rag.agent.schema import Decision

    groups = registry["groups"]
    group_ids = ["masters-thesis-v1", "violation-reference-v1", "doctoral-defense-sequence-v1"]
    pair_ids = [
        "masters-restart-follow-v1",
        "masters-no-third-defense-stop-v1",
        "violation-follow-20-v1",
        "doctoral-follow-41-v1",
        "doctoral-follow-3-v1",
        "doctoral-follow-40-v1",
    ]
    if registry["schema_version"] != 1 or [group["group_id"] for group in groups] != group_ids:
        raise ValueError("invalid prompt group registry")
    if [len(group["pairs"]) for group in groups] != [2, 1, 3]:
        raise ValueError("incomplete example groups")
    if [pair["pair_id"] for group in groups for pair in group["pairs"]] != pair_ids:
        raise ValueError("invalid stable example pair IDs")
    example_messages: list[MessageRecord] = []
    for group in groups:
        stops = 0
        for pair in group["pairs"]:
            decision = Decision.model_validate_json(pair["assistant_content"])
            stops += decision.stop
            example_messages.extend(
                [
                    MessageRecord(role="human", content=pair["human_content"]),
                    MessageRecord(role="ai", content=pair["assistant_content"]),
                ]
            )
        composition = {"FOLLOW": len(group["pairs"]) - stops, "STOP": stops}
        if group["action_composition"] != composition:
            raise ValueError("incorrect registry action composition")
    if [group["action_composition"] for group in groups] != [
        {"FOLLOW": 1, "STOP": 1},
        {"FOLLOW": 1, "STOP": 0},
        {"FOLLOW": 3, "STOP": 0},
    ]:
        raise ValueError("prompt action composition changed")
    expected_variants = [
        {
            "schema_version": 1,
            "variant_id": VARIANTS[0],
            "included_group_ids": group_ids,
            "excluded_group_ids": [],
        },
        {
            "schema_version": 1,
            "variant_id": VARIANTS[1],
            "included_group_ids": group_ids[:2],
            "excluded_group_ids": group_ids[2:],
        },
    ]
    if registry["variants"] != expected_variants:
        raise ValueError("invalid prompt variants")
    for reference in references:
        if (
            len(reference.effective_messages) != 14
            or reference.effective_messages[1:-1] != example_messages
        ):
            raise ValueError("registry differs from the frozen literal examples")
    return expected_variants


def validate_reference_arm_schedule(arms: list[Any]) -> None:
    """Kiểm tra bản sao arm A: hai slot strata và orientation cân bằng."""
    controls = [arm for arm in arms if arm.matched_pair_id is None]
    if len(controls) != 1:
        raise ValueError("intervention A requires exactly one original control arm")
    treatments = [arm for arm in arms if arm.matched_pair_id is not None]
    if len(treatments) < 2 or len(treatments) % 2 != 0:
        raise ValueError("intervention A requires an even number of treatment arms")
    strata: dict[tuple[int, ...], list[Any]] = {}
    for arm in treatments:
        positions = tuple(arm.pair_positions)
        if len(positions) != 2 or positions[1] - positions[0] != 1:
            raise ValueError(f"arm {arm.arm_id} must keep the Article 3/41 pair adjacent")
        strata.setdefault(positions, []).append(arm)
    if len(strata) != 2:
        raise ValueError("intervention A requires exactly two matched slot strata")
    for positions, members in sorted(strata.items()):
        if len(members) != 2:
            raise ValueError(f"slot stratum {positions} must contain exactly two treatment arms")
        orientations = sorted(
            member.relative_pair_order
            for member in members
            if member.relative_pair_order is not None
        )
        if orientations != ["3-41", "41-3"]:
            raise ValueError(
                f"slot stratum {positions} must contain one orientation of each pair order"
            )
    orders = [tuple(arm.presented_candidates) for arm in treatments]
    if len(set(orders)) != len(orders):
        raise ValueError("intervention A requires unique presented candidate orders")


def validate_sources(
    spec: PromptAblationSpec,
    cases: list[Any],
    snapshot: Any,
    references: list[Any],
    snapshot_hash: str,
) -> list[Any]:
    """Kiểm tra roster mười input, nhãn, snapshot và liên kết lịch A đã sao."""
    reference_arms: ReferenceArmSet = spec.reference_arms
    validate_reference_arm_schedule(reference_arms.arms)
    if snapshot.snapshot_id != cases[0].snapshot_id:
        raise ValueError("snapshot identity mismatch")
    if sorted(case.question_id for case in cases) != list(range(1, 9)):
        raise ValueError("requires complete fixed Q1-Q8 case roster")
    eligible = [case for case in cases if case.candidates]
    if [case.question_id for case in eligible] != [1, 3, 4, 5, 6, 7]:
        raise ValueError("eligible roster differs")
    for case in cases:
        if case.snapshot_hash != snapshot_hash or case.snapshot_id != snapshot.snapshot_id:
            raise ValueError("case snapshot provenance differs")
        if (
            case.dataset_id != snapshot.dataset_id
            or sha256_text(case.observation) != case.observation_hash
        ):
            raise ValueError("case source identity differs")
        if case.candidates and (
            case.label_status != "approved" or case.expected_action not in {"follow", "stop"}
        ):
            raise ValueError("eligible cases require unchanged approved labels")
    q5 = next(case for case in eligible if case.question_id == 5)
    if (
        source_identity(q5) != reference_arms.source_case_identity
        or q5.candidates != reference_arms.grammar_candidates
    ):
        raise ValueError("A source case or fixed grammar differs")
    roster: dict[tuple[str, tuple[int, ...]], Any] = {}
    for case in eligible:
        orders = [case.candidates]
        if case.question_id == 5:
            orders += [arm.presented_candidates for arm in reference_arms.arms]
        for order in orders:
            roster[(case.case_id, tuple(order))] = case
    if len(references) != 10 or len({reference.reference_id for reference in references}) != 10:
        raise ValueError("requires ten unique baseline references")
    if {
        (reference.source_case_identity.case_id, tuple(reference.presented_candidates))
        for reference in references
    } != set(roster):
        raise ValueError("baseline reference roster differs")
    for reference in references:
        case = roster[(reference.source_case_identity.case_id, tuple(reference.presented_candidates))]
        if (
            reference.source_case_identity != source_identity(case)
            or reference.question != case.question
            or reference.observation != case.observation
            or reference.grammar_candidates != case.candidates
        ):
            raise ValueError("baseline reference final input differs from frozen case")
        key = structured_hash(
            [
                reference.source_case_identity.model_dump(),
                reference.question,
                reference.observation,
                reference.presented_candidates,
                reference.grammar_text,
            ]
        )
        if key != reference.reference_id:
            raise ValueError("baseline reference identity mismatch")
    return eligible


def build_prompt_input_traces(
    references: list[Any],
    config: Any,
) -> list[PromptAblationInputTrace]:
    """Tạo input identity từ evidence bất biến, không phụ thuộc treatment."""
    traces: list[PromptAblationInputTrace] = []
    for reference in references:
        data = {
            key: getattr(reference, key)
            for key in (
                "reference_id",
                "source_case_identity",
                "question",
                "observation",
                "presented_candidates",
                "grammar_candidates",
                "grammar_text",
                "grammar_hash",
            )
        }
        data.update(
            case_id=reference.source_case_identity.case_id,
            question_hash=sha256_text(reference.question),
            observation_hash=sha256_text(reference.observation),
            execution_config_hash=structured_hash(config.model_dump(mode="json")),
        )
        identity = structured_hash(
            {**data, "source_case_identity": reference.source_case_identity.model_dump()}
        )
        traces.append(PromptAblationInputTrace(input_trace_id=identity, **data))
    return traces


def build_prompt_traces(
    input_traces: list[PromptAblationInputTrace],
    references: list[Any],
    registry: dict[str, Any],
    config: Any,
    *,
    render: bool = False,
) -> list[PromptTrace]:
    """Dựng lại trace từ bằng chứng đóng băng; render chỉ kiểm tra ở prepare."""
    variants = validate_registry(registry, references)
    traces: list[PromptTrace] = []
    inputs = {trace.reference_id: trace for trace in input_traces}
    for reference in references:
        source = inputs[reference.reference_id]
        for variant in variants:
            if variant["variant_id"] == VARIANTS[0]:
                messages = reference.effective_messages
                if render:
                    from src.rag.agent.prompt import build_decision_grammar, render_gate_prompt

                    actual = render_gate_prompt(
                        reference.question, reference.observation, reference.presented_candidates
                    )
                    actual_messages = [
                        MessageRecord(role=message.type, content=message.content)
                        for message in actual.to_messages()
                    ]
                    if (
                        actual_messages != messages
                        or build_decision_grammar(reference.grammar_candidates)
                        != reference.grammar_text
                    ):
                        raise ValueError("production baseline parity failed")
            else:
                messages = [reference.effective_messages[0]]
                messages.extend(
                    MessageRecord(role=row["role"], content=row["content"])
                    for row in _registry_messages(registry, variant["included_group_ids"])
                )
                messages.append(reference.effective_messages[-1])
            identity = prompt_request_identity(messages, variant)
            fingerprint = compute_request_fingerprint(
                reference.source_case_identity,
                messages,
                reference.presented_candidates,
                reference.grammar_text,
                config,
            )
            prompt_id = structured_hash(
                [source.input_trace_id, identity.model_dump(), fingerprint]
            )
            traces.append(
                PromptTrace(
                    prompt_trace_id=prompt_id,
                    input_trace_id=source.input_trace_id,
                    grammar_candidates=reference.grammar_candidates,
                    grammar_text=reference.grammar_text,
                    grammar_hash=reference.grammar_hash,
                    effective_messages=messages,
                    effective_messages_hash=identity.effective_messages_hash,
                    request_fingerprint=fingerprint,
                    prompt_variant_id=variant["variant_id"],
                    prompt_identity=identity,
                    baseline_parity=True if variant["variant_id"] == VARIANTS[0] else None,
                )
            )
    return traces


def plan_prompt_ablation(
    spec: PromptAblationSpec,
    cases: list[Any],
    input_traces: list[PromptAblationInputTrace],
    traces: list[PromptTrace],
) -> list[PromptAblationTrial]:
    """Dựng phase/batch/repeat/input/treatment order không cần kết quả cũ."""
    eligible = [case for case in cases if case.candidates]
    q5 = next(case for case in eligible if case.question_id == 5)
    arms = {arm.arm_id: arm for arm in spec.reference_arms.arms}
    input_lookup = {(trace.case_id, tuple(trace.presented_candidates)): trace for trace in input_traces}
    lookup = {(trace.input_trace_id, trace.prompt_variant_id): trace for trace in traces}
    trials: list[PromptAblationTrial] = []
    for phase in spec.phases:
        for batch_index, treatment_order in enumerate(spec.batches):
            if phase == PHASES[0]:
                inputs = [
                    (q5, arm_id, arms[arm_id].presented_candidates)
                    for arm_id in spec.reference_arms.batches[batch_index].arm_order
                ]
            else:
                case_order = eligible if batch_index == 0 else eligible[::-1]
                inputs = [(case, "original", case.candidates) for case in case_order]
            for repeat in range(3):
                for case, arm_id, order in inputs:
                    source = input_lookup[(case.case_id, tuple(order))]
                    pair_id = f"{phase}:{case.case_id}:{arm_id}:batch-{batch_index}:r{repeat}"
                    for variant in treatment_order:
                        trace = lookup[(source.input_trace_id, variant)]
                        trials.append(
                            PromptAblationTrial(
                                trial_id=f"{pair_id}:{variant}",
                                input_trace_id=trace.input_trace_id,
                                prompt_trace_id=trace.prompt_trace_id,
                                case_id=case.case_id,
                                phase_id=phase,
                                arm_id=arm_id,
                                batch_id=f"batch-{batch_index}",
                                repeat_id=repeat,
                                prompt_variant_id=variant,
                                prompt_treatment_pair_id=pair_id,
                                effective_messages_hash=trace.effective_messages_hash,
                                included_example_group_ids=trace.prompt_identity.included_group_ids,
                                excluded_example_group_ids=trace.prompt_identity.excluded_group_ids,
                                source_a_arm_id=arm_id if phase == PHASES[0] else None,
                                case_role=ROLES[str(case.question_id)],
                                presented_candidates=order,
                                grammar_candidates=trace.grammar_candidates,
                                observation_hash=source.observation_hash,
                                grammar_text=source.grammar_text,
                                grammar_hash=trace.grammar_hash,
                                request_fingerprint=trace.request_fingerprint,
                            )
                        )
    return trials


__all__ = [
    "build_prompt_definition",
    "build_prompt_input_traces",
    "build_prompt_traces",
    "plan_prompt_ablation",
    "prompt_request_identity",
    "source_identity",
    "validate_reference_arm_schedule",
    "validate_registry",
    "validate_sources",
]
