"""
    Streamlit-side HTTP client cho chat turn: POST /conversations/{id}/messages
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from frontend.services.http_config import API_BASE_URL, auth_headers, chat_timeout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResult:
    """Kết quả happy-path đã parse từ ChatResponse của backend"""

    answer: str
    sources: list[dict]
    memory_persisted: bool


def send_message(
    conversation_id: str,
    user_id: str,
    question: str,
    *,
    client: httpx.Client | None = None,
) -> ChatResult | None:
    """Gửi một chat turn tới backend; ChatResult nếu 200, None nếu request fail.

    Seam `client=` cho test: bỏ qua thì hàm tự sở hữu + đóng httpx.Client (timeout
    chat dài); truyền vào thì mượn, KHÔNG đóng client của caller.
    """
    owns_client = client is None
    http = client if client is not None else httpx.Client(timeout=chat_timeout())
    try:
        response = http.post(
            f"{API_BASE_URL}/conversations/{conversation_id}/messages",
            headers=auth_headers(user_id),
            json={"question": question},
        )
        if response.status_code == 404:
            logger.warning(
                "Conversation %s trả 404 cho user %s - không tồn tại hoặc "
                "không phải của user này",
                conversation_id,
                user_id,
            )
            return None
        response.raise_for_status()
        payload = response.json()
        # payload["answer"] cố ý hard-access: answer là bắt buộc, thiếu -> KeyError
        # (fail-loud). sources/memory_persisted optional với default an toàn.
        return ChatResult(
            answer=payload["answer"],
            sources=payload.get("sources") or [],
            memory_persisted=bool(payload.get("memory_persisted", False)),
        )
    except httpx.HTTPError:
        logger.exception(
            "Failed to send message to conversation %s", conversation_id
        )
        return None
    finally:
        if owns_client:
            http.close()
