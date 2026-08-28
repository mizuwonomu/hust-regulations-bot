"""Route tests cho POST /conversations/{id}/messages/stream.

Ba biên của contract "error = exception trước stream, event sau stream":
- happy: sequence sources (đúng 1) -> token*N -> done, memory_persisted đọc từ
  hộp thư sau khi loop drain (ghi history thật vào test DB qua bind_history).
- owner mismatch: 404 thật + KHÔNG event nào (stream chưa mở).
- chain nổ giữa astream: các token trước đó giữ nguyên, stream kết thúc bằng
  error event chứ không 500 (200 đã commit, raise nữa là vỡ wire).

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

from src.api.dependencies import get_rag_chain, get_sync_db_pool
from src.api.routes.chat_stream import router as chat_stream_router

USER = "user_stream"
OTHER = "user_other"


def _cid() -> str:
    return str(uuid.uuid4())


def make_fake_stream_chain(chunks: list[dict], fail_after: int | None = None):
    """Stub stream: async generator nhả chunk canned, bọc trong RunnableLambda.

    BẮT BUỘC đi qua RunnableLambda (hoặc composite thật): nó tự tạo callback
    runs nên on_end listener (_aexit_history) của RunnableWithMessageHistory
    chạy và hộp thư được ghi cờ. Custom Runnable override astream thủ công
    KHÔNG có callback machinery - listener bị nuốt, memory_persisted luôn
    false. Đây là bài học của spike, không phải bug route
    """

    async def _gen(inputs: dict):
        
        for emitted, chunk in enumerate(chunks):
            if fail_after is not None and emitted >= fail_after:
                raise RuntimeError("Stub lỗi giữa astream.")
            emitted += 1
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
    (test DB thật - history read/write qua bind_history chạy thật) + fake chain"""

    clients: list[httpx.AsyncClient] = []

    async def _make(chunks: list[dict], fail_after: int | None = None):
        app = FastAPI()
        app.include_router(chat_stream_router)
        app.dependency_overrides[get_sync_db_pool] = lambda: test_pool
        app.dependency_overrides[get_rag_chain] = lambda: make_fake_stream_chain(chunks, fail_after)
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
async def test_happy_path_sources_once_then_tokens_then_done(make_stream_client):
    """Sequence đúng contract: 1 sources -> token*2 -> done(true); sources
    đến TRƯỚC token (retrieval xong trước gen) chứ không dồn về cuối"""
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


@pytest.mark.anyio
async def test_owner_mismatch_is_404_with_no_events(make_stream_client, seed_conversation):
    """Conversation của user khác -> 404 thật (stream chưa mở, lỗi vẫn là
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


@pytest.mark.anyio
async def test_chain_failure_mid_stream_ends_with_error_event(make_stream_client):
    """Nổ giữa astream: token trước đó vẫn đã đến client, frame cuối là
    error event, KHÔNG có done và KHÔNG 500"""
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
