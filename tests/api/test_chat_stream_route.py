"""Route tests cho POST /conversations/{id}/messages/stream.

Các biên của contract:

- B1 history: route đọc history từ DB và inject vào input của astream.
- B2 happy: sequence sources (đúng 1) -> token*N -> done(true), answer ghép
  từ delta phải nằm trong DB (1 cặp question/answer).
- B3 write fail: persist_turn nổ -> vẫn emit đủ token, frame cuối done(false),
  KHÔNG row nào trong DB.
- B4 chain nổ giữa astream: token trước đó giữ nguyên, frame cuối là error
  event (không 500, không done), bước write không bao giờ chạy -> KHÔNG row.
- B5 owner mismatch: 404 thật + KHÔNG event nào (stream chưa mở).

Lưu ý: user_id trên cột conversations là varchar(15) - constant phải <= 15 ký tự.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.runnables import RunnableLambda

import src.api.routes.chat_stream as chat_stream_module
from src.api.dependencies import get_rag_chain, get_sync_db_pool
from src.api.routes.chat_stream import router as chat_stream_router

USER = "user_stream"
OTHER = "user_other"


def _cid() -> str:
    return str(uuid.uuid4())


def make_fake_stream_chain(chunks: list[dict], fail_after: int | None = None, record: dict | None = None):
    """Stub stream: async generator nhả chunk canned, bọc trong RunnableLambda.

    Memory không còn chạy qua callback của wrapper nên stub không cần callback
    machinery nữa - nó chỉ cần nhận đúng input mà route đẩy vào (question +
    chat_history) để test chứng minh route tự đọc và inject history. record
    (nếu có) bắt lại input đó để test khẳng định ở ngoài
    """

    async def _gen(inputs: dict):
        if record is not None:
            record["chat_history"] = inputs.get("chat_history")

        for emitted, chunk in enumerate(chunks):
            if fail_after is not None and emitted >= fail_after:
                raise RuntimeError("Stub lỗi giữa astream.")
            yield chunk

    return RunnableLambda(_gen)


HAPPY_CHUNKS = [
    {"context": [SimpleNamespace(page_content="nội dung cha", metadata={"title": "Điều 5", "doc_id": "d1"})]},
    {"answer": "Xin "},
    {"answer": "chào"},
]

DOC_KEYS = ["title", "content", "doc_id"]


@pytest.fixture
async def make_stream_client(test_pool):
    """Factory dựng app hermetic chỉ mount chat_stream router, override pool
    (test DB thật - read_history/persist_turn chạy thật) + fake chain"""

    clients: list[httpx.AsyncClient] = []

    async def _make(chunks: list[dict], fail_after: int | None = None, record: dict | None = None):
        app = FastAPI()
        app.include_router(chat_stream_router)
        app.dependency_overrides[get_sync_db_pool] = lambda: test_pool
        app.dependency_overrides[get_rag_chain] = lambda: make_fake_stream_chain(chunks, fail_after, record)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        )
        await client.__aenter__()
        clients.append(client)
        return client

    yield _make

    for client in clients:
        await client.__aexit__(None, None, None)


async def _read_sse(response: httpx.Response) -> list[tuple[str | None, dict | None]]:
    """Parse wire thành list (event_name, data_dict) theo thứ tự đến;
    bỏ qua comment (dòng bắt đầu ':') - ping của sse-starlette rơi vào đây"""
    events: list[tuple[str | None, dict | None]] = []
    name: str | None = None
    data_lines: list[str] = []

    async for line in response.aiter_lines():
        if line.startswith(":"):
            continue
        if line == "":
            if name is not None or data_lines:
                events.append((name, json.loads("\n".join(data_lines)) if data_lines else None))
            name, data_lines = None, []
            continue
        if line.startswith("event:"):
            name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())

    if name is not None or data_lines:
        events.append((name, json.loads("\n".join(data_lines)) if data_lines else None))
    return events


@pytest.mark.anyio
async def test_history_is_read_and_injected_into_astream(make_stream_client, seed_conversation):
    """B1: route đọc history từ DB (không qua wrapper) và đưa vào input của
    astream - stub nhận chat_history là list BaseMessage từ seed"""
    cid = _cid()
    seed_conversation(cid, USER, n_pairs=1)
    record: dict = {}
    client = await make_stream_client(HAPPY_CHUNKS, record=record)

    async with client.stream(
        "POST",
        f"/conversations/{cid}/messages/stream",
        headers={"X-User-Id": USER},
        json={"question": "câu mới"},
    ) as response:
        assert response.status_code == 200
        await _read_sse(response)

    assert len(record["chat_history"]) == 2
    assert [m.content for m in record["chat_history"]] == ["câu hỏi 0", "trả lời 0"]


@pytest.mark.anyio
async def test_happy_path_sources_once_then_tokens_then_done(
    make_stream_client, fetch_messages
):
    """B2: sequence đúng contract - 1 sources -> token*2 -> done(true); answer
    ghép từ delta ("Xin " + "chào") phải là row thật trong DB"""
    cid = _cid()
    client = await make_stream_client(HAPPY_CHUNKS)

    async with client.stream(
        "POST",
        f"/conversations/{cid}/messages/stream",
        headers={"X-User-Id": USER},
        json={"question": "điều kiện tốt nghiệp?"},
    ) as response:
        assert response.status_code == 200
        events = await _read_sse(response)

    names = [name for name, _ in events]
    assert names == ["sources", "token", "token", "done"]

    sources_data = events[0][1]
    assert [list(s.keys()) for s in sources_data["sources"]] == [DOC_KEYS]
    assert sources_data["sources"][0]["title"] == "Điều 5"

    joined = "".join(data["text"] for name, data in events if name == "token")
    assert joined == "Xin chào"

    assert events[-1][1] == {"memory_persisted": True}

    assert await fetch_messages(cid) == [
        {"role": "user", "content": "điều kiện tốt nghiệp?"},
        {"role": "ai", "content": "Xin chào"},
    ]


@pytest.mark.anyio
async def test_persist_failure_still_done_false_and_no_row(
    make_stream_client, monkeypatch, fetch_messages
):
    """B3: persist_turn nổ -> mọi token vẫn emit, frame cuối là done(false)
    (lỗi memory không vứt answer, không lên wire), DB không có row"""
    cid = _cid()

    def _boom(*args, **kwargs):
        raise RuntimeError("DB sập lúc ghi")

    monkeypatch.setattr(chat_stream_module, "persist_turn", _boom)
    client = await make_stream_client(HAPPY_CHUNKS)

    async with client.stream(
        "POST",
        f"/conversations/{cid}/messages/stream",
        headers={"X-User-Id": USER},
        json={"question": "câu hỏi"},
    ) as response:
        assert response.status_code == 200
        events = await _read_sse(response)

    names = [name for name, _ in events]
    assert names == ["sources", "token", "token", "done"]
    assert events[-1][1] == {"memory_persisted": False}

    joined = "".join(data["text"] for name, data in events if name == "token")
    assert joined == "Xin chào"

    assert await fetch_messages(cid) == []


@pytest.mark.anyio
async def test_chain_failure_mid_stream_ends_with_error_event_and_no_row(
    make_stream_client, fetch_messages
):
    """B4: nổ giữa astream - token trước đó vẫn đã đến client, frame cuối là
    error event, KHÔNG done và KHÔNG 500; bước write không bao giờ chạy"""
    cid = _cid()
    client = await make_stream_client(HAPPY_CHUNKS, fail_after=2)

    async with client.stream(
        "POST",
        f"/conversations/{cid}/messages/stream",
        headers={"X-User-Id": USER},
        json={"question": "câu hỏi nổ"},
    ) as response:
        assert response.status_code == 200
        events = await _read_sse(response)

    names = [name for name, _ in events]
    assert names[:2] == ["sources", "token"]
    assert names[-1] == "error"
    assert "done" not in names
    assert isinstance(events[-1][1]["message"], str) and events[-1][1]["message"]

    assert await fetch_messages(cid) == []


@pytest.mark.anyio
async def test_owner_mismatch_is_404_with_no_events(make_stream_client, seed_conversation):
    """B5: conversation của user khác -> 404 thật (stream chưa mở, lỗi vẫn là
    exception) và wire không có event nào bị lộ"""
    cid = _cid()
    seed_conversation(cid, OTHER)
    client = await make_stream_client(HAPPY_CHUNKS)

    response = await client.post(
        f"/conversations/{cid}/messages/stream",
        headers={"X-User-Id": USER},
        json={"question": "xin chào"},
    )

    assert response.status_code == 404
    assert cid not in response.text
    assert "text/event-stream" not in response.headers.get("content-type", "")
