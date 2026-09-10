"""Adapter that sends one initial-gate decision through the existing gate."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from src.rag.agent.gate import decide_next
from src.rag.agent.llm_client import create_llm_client
from src.rag.agent.schema import Decision


def decide(
    question: str,
    observation: str,
    candidates: list[int],
    *,
    client: ChatOpenAI,
) -> Decision:
    """Call the unchanged gate once with the supplied client."""
    if not isinstance(question, str) or not isinstance(observation, str):
        raise ValueError("question and observation must be strings")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    if any(isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0 for candidate in candidates):
        raise ValueError("candidates must contain positive integers")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must not contain duplicates")
    return decide_next(question, observation, candidates, client=client)


def create_client() -> ChatOpenAI:
    """Create the existing local gate client lazily for one run."""
    return create_llm_client()


def _json_setting(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return None


def client_config(client: Any | None = None) -> dict[str, Any]:
    """Return non-secret effective client settings available to the harness."""
    if client is None:
        return {
            "model_alias": None,
            "actual_model": None,
            "actual_model_reason": "client is not initialized",
            "temperature": None,
            "sampling": {"temperature": None},
            "thinking": None,
            "token_limit": None,
            "timeout_seconds": None,
            "retry_settings": None,
        }

    extra_body = getattr(client, "extra_body", None)
    thinking = None
    if isinstance(extra_body, dict):
        chat_template_kwargs = extra_body.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict):
            thinking = chat_template_kwargs.get("enable_thinking")

    temperature = _json_setting(getattr(client, "temperature", None))
    timeout = getattr(client, "timeout", None)
    if timeout is None:
        timeout = getattr(client, "request_timeout", None)
    return {
        "model_alias": _json_setting(getattr(client, "model", None)),
        "actual_model": None,
        "actual_model_reason": "local server model is not exposed by the client",
        "temperature": temperature,
        "sampling": {"temperature": temperature},
        "thinking": _json_setting(thinking),
        "token_limit": _json_setting(getattr(client, "max_completion_tokens", None)),
        "timeout_seconds": _json_setting(timeout),
        "retry_settings": _json_setting(getattr(client, "max_retries", None)),
    }
