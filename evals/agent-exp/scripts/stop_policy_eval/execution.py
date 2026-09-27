"""Render request baseline, chạy một invocation và băm request của STOP replay

Module chỉ dựng request từ `GateInput` không nhãn, gọi đúng một lần production
gate cho mỗi trial và giữ nguyên classification lỗi dùng chung với agent-exp
"""

from __future__ import annotations

from typing import Any

from contracts import DecisionOutcome

from diagnostic_subexp.shared.contracts import (
    DiagnosticExecutionConfig,
    DiagnosticGateRequest,
    MessageRecord,
    sha256_text,
    structured_hash,
)
from diagnostic_subexp.shared.execution import (
    build_effective_messages,
    build_grammar_text,
    effective_client_config,
    execute_diagnostic_request,
)

from stop_policy_eval.cases import label_free_input
from stop_policy_eval.contracts import (
    StopInputTrace,
    StopPolicyCase,
    hop_class_for,
    resolve_variant,
)

VALID_POLICIES: tuple[str, ...] = ("first", "llm")
PREPARED_CONFIG_REASON = "the local gate client is created when the run starts"


def effective_execution_config(client: Any) -> DiagnosticExecutionConfig:
    """Đọc cấu hình non-secret hiệu lực của client local trước lời gọi đầu tiên"""
    raw = effective_client_config(client)
    required = (
        "model_alias",
        "base_url",
        "temperature",
        "max_completion_tokens",
        "enable_thinking",
    )
    missing = [name for name in required if raw.get(name) is None]
    if missing:
        raise ValueError(f"the local gate client does not expose: {missing}")
    return DiagnosticExecutionConfig(
        model_alias=str(raw["model_alias"]),
        base_url=str(raw["base_url"]),
        temperature=float(raw["temperature"]),
        max_completion_tokens=int(raw["max_completion_tokens"]),
        enable_thinking=bool(raw["enable_thinking"]),
        timeout_seconds=(
            None if raw.get("timeout_seconds") is None else float(raw["timeout_seconds"])
        ),
        max_retries=raw.get("max_retries"),
        extra_request_options=raw.get("extra_request_options") or {},
    )


def compute_input_fingerprint(
    *,
    input_trace_id: str,
    variant: str,
    case: StopPolicyCase,
    effective_messages: list[MessageRecord],
    grammar_text: str,
) -> str:
    """Băm request baseline đầy đủ, bỏ mọi yếu tố phụ thuộc repeat hay kết quả"""
    return structured_hash(
        {
            "input_trace_id": input_trace_id,
            "variant": variant,
            "dataset_id": case.dataset_id,
            "question_id": case.question_id,
            "case_id": case.case_id,
            "source_hop": case.source_hop,
            "question": case.question,
            "observation": case.observation,
            "candidates": list(case.candidates),
            "effective_messages": [
                message.model_dump(mode="json") for message in effective_messages
            ],
            "grammar_text": grammar_text,
        }
    )


def render_baseline_trace(case: StopPolicyCase, variant: str) -> StopInputTrace:
    """Render trace baseline cho một case bằng đúng production renderer

    Trace là repeat-independent: mọi repeat của cùng một case dùng chung trace và
    fingerprint này
    """
    canonical_variant = resolve_variant(variant)
    source_input = label_free_input(case)
    messages = build_effective_messages(
        source_input.question,
        source_input.observation,
        list(source_input.candidates),
    )
    grammar_text = build_grammar_text(list(source_input.candidates))
    input_trace_id = f"{case.case_id}:{canonical_variant}"
    return StopInputTrace(
        input_trace_id=input_trace_id,
        case_id=case.case_id,
        dataset_id=case.dataset_id,
        question_id=case.question_id,
        source_hop=case.source_hop,
        hop_class=hop_class_for(case.source_hop),
        variant=canonical_variant,
        question=source_input.question,
        observation=source_input.observation,
        question_hash=sha256_text(source_input.question),
        observation_hash=sha256_text(source_input.observation),
        candidates=list(source_input.candidates),
        grammar_candidates=list(source_input.candidates),
        effective_messages=messages,
        messages_hash=structured_hash(
            [message.model_dump(mode="json") for message in messages]
        ),
        grammar_text=grammar_text,
        grammar_hash=sha256_text(grammar_text),
        input_fingerprint=compute_input_fingerprint(
            input_trace_id=input_trace_id,
            variant=canonical_variant,
            case=case,
            effective_messages=messages,
            grammar_text=grammar_text,
        ),
    )


def request_fingerprint(
    trace: StopInputTrace,
    policy: str,
    execution_config: DiagnosticExecutionConfig | None,
) -> str:
    """Băm request thực thi của một policy; chỉ llm mới mang cấu hình client"""
    if policy not in VALID_POLICIES:
        raise ValueError(f"unsupported policy: {policy!r}")
    payload: dict[str, Any] = {
        "policy": policy,
        "input_fingerprint": trace.input_fingerprint,
    }
    if policy == "llm":
        if execution_config is None:
            raise ValueError(
                "the llm policy requires a recorded execution config before fingerprinting"
            )
        payload["execution_config"] = execution_config.model_dump(mode="json")
    return structured_hash(payload)


def request_from_trace(trace: StopInputTrace) -> DiagnosticGateRequest:
    """Dựng request hiệu lực từ trace đã lưu, không cần label hay case file"""
    return DiagnosticGateRequest(
        question=trace.question,
        observation=trace.observation,
        presented_candidates=list(trace.candidates),
        grammar_candidates=list(trace.grammar_candidates),
        effective_messages=list(trace.effective_messages),
        grammar_text=trace.grammar_text,
    )


def execute_stop_request(
    trace: StopInputTrace,
    *,
    policy: str,
    client: Any | None = None,
) -> DecisionOutcome:
    """Chạy đúng một invocation cho một policy, trả outcome đã phân loại lỗi"""
    return execute_diagnostic_request(request_from_trace(trace), policy=policy, client=client)


__all__ = [
    "PREPARED_CONFIG_REASON",
    "VALID_POLICIES",
    "compute_input_fingerprint",
    "effective_execution_config",
    "execute_stop_request",
    "render_baseline_trace",
    "request_fingerprint",
    "request_from_trace",
]
