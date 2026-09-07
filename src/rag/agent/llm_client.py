"""Module gọi LLM local, serve client qua OpenAI endpoint."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.rag.config import (
    CITATION_AGENT_BASE_URL,
    CITATION_AGENT_MAX_TOKENS,
    CITATION_AGENT_TEMPERATURE,
)


def create_llm_client(
    *,
    base_url: str = CITATION_AGENT_BASE_URL,
    temperature: float = CITATION_AGENT_TEMPERATURE,
    max_completion_tokens: int = CITATION_AGENT_MAX_TOKENS,
) -> ChatOpenAI:
    """Dựng HTTP client tới llama-server đã chạy sẵn.

    Params:
    - base_url: Endpoint OpenAI-compatible của llama-server local.
    - temperature: Temperature cho gate.
    - max_completion_tokens: Giới hạn token output của một quyết định gate.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url phải là chuỗi HTTP(S) không rỗng")
    base_url = base_url.strip()
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(f"base_url phải là HTTP(S): {base_url!r}")

    return ChatOpenAI(
        base_url=base_url,
        model="citation-agent",
        api_key="dummy",
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )
