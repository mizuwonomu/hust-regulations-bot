"""Unit tests cho frontend SSE client.

MockTransport nhả canned SSE body - không live server, không DB. Contract hai
chiều y hệt test_chat_client: request ra đúng shape (method/path/header/body)
+ response 200 parse vào iterator token + side-channel (sources,
memory_persisted, error). 

Convention fail: 404 / transport / non-200 -> None
(no-answer, hard-fail ở UI); error EVENT -> không raise raw, client ghi cờ
error và dừng - caller quyết định hard/partial dựa trên token đã yield.
"""

from __future__ import annotations

import json

import httpx

from frontend.services.chat_stream_client import stream_message

CONV_ID = "conv-stream-1"
USER_ID = "user_vjp_pro_1"
QUESTION = "điều kiện tốt nghiệp là gì?"

SOURCE = {"title": "Điều 5", "content": "nội dung", "doc_id": "d1"}


def _sse(*frames: tuple[str, dict]) -> str:
    """Dựng wire body SSE từ các cặp (event_name, data_dict)"""
    return "".join(
        f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        for name, data in frames
    )


HAPPY_BODY = _sse(
    ("sources", {"sources": [SOURCE]}),
    ("token", {"text": "Xin "}),
    ("token", {"text": "chào"}),
    ("done", {"memory_persisted": True}),
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_happy_path_sends_well_formed_request_and_parses_stream():
    """Một test, hai chiều contract: request ra đúng shape + SSE 200 parse vào"""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["user"] = request.headers.get("x-user-id")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=HAPPY_BODY
        )

    stream = stream_message(CONV_ID, USER_ID, QUESTION, client=_client(handler))
    assert stream is not None

    joined = "".join(stream)
    assert joined == "Xin chào"
    assert seen == {
        "method": "POST",
        "path": f"/conversations/{CONV_ID}/messages/stream",
        "user": USER_ID,
        "body": {"question": QUESTION},
    }
    assert stream.sources == [SOURCE]
    assert stream.memory_persisted is True
    assert stream.error is None


def test_sources_side_channel_available_during_iteration():
    """Side-channel đọc được GIỮA iteration - sources frame đến trước token,
    không phải chờ stream drain xong mới thấy"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=HAPPY_BODY
        )

    stream = stream_message(CONV_ID, USER_ID, QUESTION, client=_client(handler))
    iterator = iter(stream)
    assert next(iterator) == "Xin "
    assert stream.sources == [SOURCE]


def test_error_event_reported_not_raised():
    """Error event giữa stream: token trước đó vẫn yield đủ, client ghi message,
    KHÔNG raise raw và không có done sau đó"""
    body = _sse(
        ("token", {"text": "một nửa "}),
        ("error", {"message": "chain gãy giữa stream"}),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )

    stream = stream_message(CONV_ID, USER_ID, QUESTION, client=_client(handler))
    joined = "".join(stream)

    assert joined == "một nửa "
    assert stream.error == "chain gãy giữa stream"
    assert stream.memory_persisted is False


def test_missing_done_defaults_memory_persisted_false():
    """Stream kết thúc đột ngột không done frame - memory_persisted phải là
    False (default an toàn), không raise"""
    body = _sse(("token", {"text": "chưa kịp xong"}))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )

    stream = stream_message(CONV_ID, USER_ID, QUESTION, client=_client(handler))
    assert "".join(stream) == "chưa kịp xong"
    assert stream.memory_persisted is False


def test_404_returns_none():
    """Conversation không tồn tại/không phải của user -> no-answer signal,
    KHÔNG tạo stream object"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    assert stream_message(CONV_ID, USER_ID, QUESTION, client=_client(handler)) is None


def test_transport_error_returns_none():
    """Không nối được backend -> no-answer signal"""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("backend chết")

    assert stream_message(CONV_ID, USER_ID, QUESTION, client=_client(handler)) is None


def test_500_returns_none():
    """Borrow-1 nổ ở server (500 pre-stream) -> no-answer signal như transport
    error, không cố parse non-SSE body"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    assert stream_message(CONV_ID, USER_ID, QUESTION, client=_client(handler)) is None


def test_borrowed_client_stays_open_after_iteration():
    """Seam client=: mượn client của caller, iteration xong KHÔNG đóng - cùng
    pattern với send_message"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=HAPPY_BODY
        )

    http = _client(handler)
    stream = stream_message(CONV_ID, USER_ID, QUESTION, client=http)
    assert stream is not None
    "".join(stream)
    assert http.is_closed is False


# ── Task 5: phân loại turn outcome - logic tách khỏi Streamlit runtime ──────

from types import SimpleNamespace

from frontend.workflows.chat_stream import classify_turn_outcome


def _outcome_stream(error):
    #classify chỉ đọc stream.error - fake tối giản
    return SimpleNamespace(error=error)


def test_outcome_complete_when_stream_clean():
    """Stream lành -> complete - kể cả memory_persisted=False chỉ là soft warning"""
    assert classify_turn_outcome(_outcome_stream(None), "câu trả lời") == "complete"


def test_outcome_partial_when_error_after_tokens():
    """Error event SAU khi đã có token -> partial: incomplete turn, KHÔNG append"""
    assert classify_turn_outcome(_outcome_stream("đứt giữa đường"), "một nửa") == "partial"


def test_outcome_no_answer_when_error_before_any_token():
    """Error event TRƯỚC token nào -> no_answer: hard fail khớp biên JSON path"""
    assert classify_turn_outcome(_outcome_stream("đứt giữa đường"), "") == "no_answer"
