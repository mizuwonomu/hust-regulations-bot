"""Unit test cho 2 helper lịch sử hội thoại (read_history / persist_turn).

Chạy trên DB test thật qua test_pool - không mock DAO: thứ cần chứng minh là
row trong Postgres deserialize về BaseMessage đúng kiểu, đúng thứ tự, append
không ghi đè và các session cách ly nhau. Helper là sync nên gọi từ test
async qua anyio.to_thread.run_sync, mỗi thao tác tự mở/đóng conn với
with test_pool.connection() - giống cách route sẽ mượn conn.
"""

from __future__ import annotations

import uuid

import anyio
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.database.history_manager import persist_turn, read_history


async def _read(test_pool, conversation_id: str) -> list:
    def _q():
        with test_pool.connection() as conn:
            return read_history(conn, conversation_id)

    return await anyio.to_thread.run_sync(_q)


async def _persist(test_pool, conversation_id: str, question: str, answer: str) -> None:
    def _q():
        with test_pool.connection() as conn:
            persist_turn(conn, conversation_id, question, answer)

    await anyio.to_thread.run_sync(_q)


@pytest.mark.anyio
async def test_read_history_empty_session_returns_empty_list(test_pool):
    # Session chưa có gì - read_history phải trả list rỗng, không raise
    cid = str(uuid.uuid4())
    assert await _read(test_pool, cid) == []


@pytest.mark.anyio
async def test_persist_turn_roundtrip_types_and_contents(test_pool):
    # 1 cặp question/answer -> đọc về đúng 2 message, đúng kiểu, đúng content
    cid = str(uuid.uuid4())
    await _persist(test_pool, cid, "câu hỏi gốc", "câu trả lời AI")

    messages = await _read(test_pool, cid)
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "câu hỏi gốc"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "câu trả lời AI"


@pytest.mark.anyio
async def test_persist_turn_appends_not_overwrites(test_pool):
    # Persist 2 lượt -> 4 message theo đúng thứ tự ghi, không mất lượt trước
    cid = str(uuid.uuid4())
    await _persist(test_pool, cid, "hỏi 1", "đáp 1")
    await _persist(test_pool, cid, "hỏi 2", "đáp 2")

    messages = await _read(test_pool, cid)
    assert [(type(m).__name__, m.content) for m in messages] == [
        ("HumanMessage", "hỏi 1"),
        ("AIMessage", "đáp 1"),
        ("HumanMessage", "hỏi 2"),
        ("AIMessage", "đáp 2"),
    ]


@pytest.mark.anyio
async def test_persist_turn_isolated_per_session(test_pool):
    # Ghi vào session A - session B vẫn rỗng: không dính history chéo session
    cid_a = str(uuid.uuid4())
    cid_b = str(uuid.uuid4())
    await _persist(test_pool, cid_a, "của A", "trả lời A")

    assert await _read(test_pool, cid_b) == []
