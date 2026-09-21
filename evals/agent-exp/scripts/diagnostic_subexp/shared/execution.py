"""Render request hiệu lực và gọi đúng một lần local gate client cho diagnostic."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from src.rag.agent.schema import Decision

from contracts import DecisionOutcome, TrialError, GateCase
from diagnostic_subexp.shared.contracts import (
    DiagnosticExecutionConfig,
    DiagnosticGateRequest,
    DiagnosticSourceIdentity,
    MessageRecord,
    sha256_text,
    structured_hash,
)
from run_experiments import _classify_exception, _sanitize_error


DIAGNOSTIC_CLIENT_SETTINGS = (
    "model_alias",
    "base_url",
    "temperature",
    "max_completion_tokens",
    "enable_thinking",
    "timeout_seconds",
    "max_retries",
    "extra_request_options",
)


def render_effective_prompt(
    question: str,
    observation: str,
    presented_candidates: list[int],
):
    """Gọi production prompt renderer với presented order của arm."""
    from src.rag.agent.prompt import render_gate_prompt
    return render_gate_prompt(question, observation, presented_candidates)


def build_effective_messages(
    question: str,
    observation: str,
    presented_candidates: list[int],
) -> list[MessageRecord]:
    """Trích ordered role/content đã render thành record bền vững."""
    prompt = render_effective_prompt(question, observation, presented_candidates)
    records: list[MessageRecord] = []
    for message in prompt.to_messages():
        role = {"system": "system", "human": "human", "ai": "ai"}.get(message.type)
        if role is None:
            raise ValueError(f"unsupported prompt message type: {message.type!r}")
        if not isinstance(message.content, str):
            raise ValueError("diagnostic messages must have plain string content")
        records.append(MessageRecord(role=role, content=message.content))
    if not records:
        raise ValueError("rendered prompt produced no messages")
    return records


def build_grammar_text(grammar_candidates: list[int]) -> str:
    """Dựng grammar bytes từ grammar candidates cố định của suite."""
    from src.rag.agent.prompt import build_decision_grammar
    return build_decision_grammar(grammar_candidates)


def build_prompt_from_records(messages: list[MessageRecord]) -> ChatPromptValue:
    """Dựng lại prompt từ message đã lưu để gửi đúng bytes đã ghi."""
    classes = {"system": SystemMessage, "human": HumanMessage, "ai": AIMessage}
    rendered = [classes[message.role](content=message.content) for message in messages]
    if not rendered:
        raise ValueError("stored messages must not be empty")
    return ChatPromptValue(messages=rendered)


def compute_request_fingerprint(
    source_case_identity: DiagnosticSourceIdentity,
    effective_messages: list[MessageRecord],
    presented_candidates: list[int],
    grammar_text: str,
    execution_config: DiagnosticExecutionConfig,
) -> str:
    """Băm toàn bộ request hiệu lực, bỏ tên arm, batch và repeat."""
    return structured_hash(
        {
            "case_identity": source_case_identity.model_dump(mode="json"),
            "effective_messages": [
                message.model_dump(mode="json") for message in effective_messages
            ],
            "presented_candidates": list(presented_candidates),
            "grammar_text": grammar_text,
            "execution_config": execution_config.model_dump(mode="json"),
        }
    )


def request_from_trace(trace: Any, case: GateCase) -> DiagnosticGateRequest:
    """Dựng request hiệu lực từ trace đã verify và case đóng băng."""
    if sha256_text(case.question) != trace.question_hash:
        raise ValueError("case question does not match the stored trace")
    transformed_observation = getattr(trace, "transformed_observation", None)
    if transformed_observation is not None:
        observation = transformed_observation
    else:
        observation = case.observation
    if sha256_text(observation) != trace.transformed_observation_hash:
        raise ValueError("case observation does not match the stored trace")
    return DiagnosticGateRequest(
        question=case.question,
        observation=observation,
        presented_candidates=list(trace.presented_candidates),
        grammar_candidates=list(trace.grammar_candidates),
        effective_messages=list(trace.effective_messages),
        grammar_text=trace.grammar_text,
    )


def _json_setting(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return None


def effective_client_config(client: Any) -> dict[str, Any]:
    """Đọc các setting non-secret có thể truy cập từ client hiệu lực."""
    extra_body = getattr(client, "extra_body", None)
    thinking = None
    if isinstance(extra_body, dict):
        chat_template_kwargs = extra_body.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict):
            thinking = chat_template_kwargs.get("enable_thinking")
    return {
        "model_alias": _json_setting(getattr(client, "model_name", None))
        or _json_setting(getattr(client, "model", None)),
        "base_url": _json_setting(getattr(client, "openai_api_base", None)),
        "temperature": _json_setting(getattr(client, "temperature", None)),
        "max_completion_tokens": _json_setting(getattr(client, "max_tokens", None)),
        "enable_thinking": _json_setting(thinking),
        "timeout_seconds": _json_setting(getattr(client, "request_timeout", None)),
        "max_retries": _json_setting(getattr(client, "max_retries", None)),
        "extra_request_options": extra_body if isinstance(extra_body, dict) else {},
    }


def verify_execution_config(
    execution_config: DiagnosticExecutionConfig,
    client: Any,
) -> dict[str, Any]:
    """So sánh setting hiệu lực với cấu hình đã ghi và abort khi drift."""
    effective = effective_client_config(client)
    recorded = execution_config.model_dump(mode="json")
    drift = {
        name: {"recorded": recorded.get(name), "effective": effective.get(name)}
        for name in DIAGNOSTIC_CLIENT_SETTINGS
        if recorded.get(name) != effective.get(name)
    }
    if drift:
        raise ValueError(f"client configuration drift before the first call: {drift}")
    return effective


def response_usage(response):
    """Chuẩn hóa usage khi client cung cấp token metadata."""
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        usage = (getattr(response, "response_metadata", None) or {}).get("token_usage")
    if not isinstance(usage, dict):
        return None
    aliases = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens",
               "total_tokens": "total_tokens"}
    return {key: usage.get(key, usage.get(alias)) for key, alias in aliases.items()}


def execute_diagnostic_request(
    request: DiagnosticGateRequest,
    *,
    policy: str,
    client: Any | None = None,
) -> DecisionOutcome:
    """Chạy một application-level invocation và giữ candidate membership."""
    started = time.perf_counter()
    if policy == "first":
        decision = Decision(stop=False, dieu=request.presented_candidates[0])
        return DecisionOutcome(
            status="ok",
            decision=decision,
            error=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=None,
        )
    if policy != "llm":
        raise ValueError(f"unsupported diagnostic policy: {policy!r}")
    if client is None:
        raise ValueError("the llm policy requires an initialized client")

    usage = None
    try:
        prompt = build_prompt_from_records(request.effective_messages)
        extra_body = dict(getattr(client, "extra_body", None) or {})
        extra_body["grammar"] = request.grammar_text
        response = client.invoke(prompt, extra_body=extra_body)
        usage = response_usage(response)
        decision = Decision.model_validate_json(response.content)
        if not decision.stop and decision.dieu not in request.presented_candidates:
            raise ValueError(
                f"gate chọn Điều {decision.dieu} ngoài candidates {request.presented_candidates}"
            )
        return DecisionOutcome(
            status="ok",
            decision=decision,
            error=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=usage,
        )
    except Exception as exc:  # noqa: BLE001
        return DecisionOutcome(
            status="error",
            decision=None,
            error=TrialError(
                category=_classify_exception(exc),
                message=_sanitize_error(exc),
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=usage,
        )


__all__ = [
    "build_effective_messages",
    "build_grammar_text",
    "compute_request_fingerprint",
    "build_prompt_from_records",
    "effective_client_config",
    "execute_diagnostic_request",
    "render_effective_prompt",
    "request_from_trace",
    "verify_execution_config",
]
