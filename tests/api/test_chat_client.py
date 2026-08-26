"""Unit tests cho frontend chat HTTP client (step 2 của cutover).

Chạy thuần với httpx.MockTransport: không live server, không DB. Assert body
theo SEMANTIC (parse lại bằng json.loads) chứ không so byte - tránh dính vào
separator nội bộ của httpx. Bao gồm cả contract fail-loud: body 200 méo phải
RAISE, không nuốt thành None.
"""

from __future__ import annotations

import json

import httpx
import pytest

from frontend.services.chat_client import ChatResult, send_message

CONV_ID = "conv-123"
USER_ID = "user_vjp_pro_1"
QUESTION = "điều kiện tốt nghiệp là gì?"

CANNED_BODY = {
    "answer": "Theo quy chế đào tạo...",
    "sources": [
        {"title": "Điều 12", "content": "...", "doc_id": "doc-1"},
    ],
    "memory_persisted": True,
}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_happy_path_sends_well_formed_request_and_parses_200():
    """Một test, hai chiều contract: request ra đúng shape + response 200 parse vào."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = request.url
        seen["user_id"] = request.headers.get("X-User-Id")
        seen["json"] = json.loads(request.read())
        return httpx.Response(200, json=CANNED_BODY)

    result = send_message(CONV_ID, USER_ID, QUESTION, client=_client(handler))

    # chiều đi
    assert seen["method"] == "POST"
    assert seen["url"].path == f"/conversations/{CONV_ID}/messages"
    assert seen["user_id"] == USER_ID
    assert seen["json"] == {"question": QUESTION}
    # chiều về
    assert isinstance(result, ChatResult)
    assert result.answer == CANNED_BODY["answer"]
    assert result.sources == CANNED_BODY["sources"]
    assert result.memory_persisted is True


def test_404_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    assert send_message(CONV_ID, USER_ID, QUESTION, client=_client(handler)) is None


def test_transport_error_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("backend down")

    assert send_message(CONV_ID, USER_ID, QUESTION, client=_client(handler)) is None


def test_memory_not_persisted_still_returns_result():
    """Soft-degrade (client half): 200 với memory_persisted=false vẫn ra ChatResult,
    KHÔNG phải None"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**CANNED_BODY, "memory_persisted": False})

    result = send_message(CONV_ID, USER_ID, QUESTION, client=_client(handler))

    assert isinstance(result, ChatResult)
    assert result.memory_persisted is False
    assert result.answer == CANNED_BODY["answer"]


def test_missing_sources_defaults_to_empty_list():
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"answer": "chitchat", "memory_persisted": True}
        return httpx.Response(200, json=body)

    result = send_message(CONV_ID, USER_ID, QUESTION, client=_client(handler))

    assert isinstance(result, ChatResult)
    assert result.sources == []


def test_malformed_200_missing_answer_raises_fail_loud():
    """Fail-loud: body 200 thiếu `answer` là vỡ contract của chính backend mình
    -> phải RAISE (KeyError), không nuốt thành None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"sources": [], "memory_persisted": True})

    with pytest.raises(KeyError):
        send_message(CONV_ID, USER_ID, QUESTION, client=_client(handler))
