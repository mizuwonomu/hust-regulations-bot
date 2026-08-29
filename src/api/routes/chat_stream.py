"""
    Route SSE cho một lượt chat dạng token stream
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg_pool import ConnectionPool, PoolTimeout
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from src.api.dependencies import get_current_user_id, get_rag_chain, get_sync_db_pool
from src.api.schemas.chat import ChatRequest
from src.api.schemas.chat_stream import DoneEvent, ErrorEvent, SourcesEvent, TokenEvent
from src.database.conversation_queries import (
    claim_conversation,
    fetch_conversation_owner,
)
from src.database.history_manager import persist_turn, read_history
from src.database.pool_config import CONN_POLL_INTERVAL, CONN_POLL_RETRY_DELAY

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["chat"])

# Thông báo chung cho client: lý do thật chỉ đi vào log server-side
STREAM_ERROR_MESSAGE = "Đã xảy ra lỗi khi tạo câu trả lời."


async def _acquire_conn(pool: ConnectionPool):
    """Mượn conn sync khỏi pool mà KHÔNG giữ thread trong lúc chờ.

    Chờ tối đa pool.timeout (semantics gốc của pool), hết giờ raise PoolTimeout
    - borrow-1 chưa mở stream nên thành 500 thật, borrow-2 đã mở stream nên
    thành error event
    """
    deadline = time.monotonic() + pool.timeout
    while True:
        try:
            return await run_in_threadpool(pool.getconn, timeout=CONN_POLL_INTERVAL) # Mỗi lần chỉ block thread = interval rồi nhả token
        except PoolTimeout:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(CONN_POLL_RETRY_DELAY)


@router.post("/{conversation_id}/messages/stream")
async def post_message_stream(
    conversation_id: str,
    payload: ChatRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    pool: Annotated[ConnectionPool, Depends(get_sync_db_pool)],
    core_chain: Annotated[object, Depends(get_rag_chain)],
) -> EventSourceResponse:
    # Borrow-1 (ngắn): ownership + eager-create. Mượn qua _acquire_conn (không
    # block loop lúc chờ), từng query là thread op ngắn, trả conn ngay ở finally
    conn = await _acquire_conn(pool)
    try:
        owner = await run_in_threadpool(fetch_conversation_owner, conn, conversation_id) # Đẩy tác vụ psycopg sync (blocking I/O) sang thread worker

        # owner khác user -> 404 chứ không 403: không lộ sự tồn tại của conversation
        # Stream chưa mở nên đây vẫn là exception -> HTTP status thật
        if owner is not None and owner != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        if owner is None:
            await run_in_threadpool(claim_conversation, conn, conversation_id, user_id)
            await run_in_threadpool(conn.commit)

        conversation_history = await run_in_threadpool(read_history, conn, conversation_id) # Borrow-read ngắn

    except Exception:
        await run_in_threadpool(conn.rollback) # Nếu có bất kì lỗi, rollback conn trong threadpool
        raise

    finally:
        await run_in_threadpool(pool.putconn, conn)

    async def event_stream() -> AsyncIterator[dict]:
        answer_parts: list[str] = []

        try:
            # Generator không giữ conn trong suốt thời gian đợi I/O Groq
            sources_sent = False
            async for chunk in core_chain.astream(
                {"question": payload.question, "chat_history": conversation_history},
                config={"configurable": {"session_id": conversation_id}},
            ):
                # Emit frame: context đến sớm (retrieval xong trước gen) -> sources là frame
                # đầu, emit NGAY không dồn về cuối; capture-once
                if "context" in chunk and not sources_sent:
                    sources_sent = True
                    yield {
                        "event": SourcesEvent.event,
                        "data": SourcesEvent.from_docs(chunk["context"]).to_json(),
                    }
                if "answer" in chunk:
                    answer_parts.append(chunk["answer"]) # Append full string messages -> lưu vào DB

                    yield {
                        "event": TokenEvent.event,
                        "data": TokenEvent(text=chunk["answer"]).to_json(),
                    }
            
        except Exception:
            # Stream đã mở (status khoá 200) - exception không còn đường raise
            # thành HTTP status
            logger.exception(
                "SSE stream lỗi giữa đường cho conversation: %s", conversation_id
            )
            yield {
                "event": ErrorEvent.event,
                "data": ErrorEvent(message=STREAM_ERROR_MESSAGE).to_json(),
            }

            return # Nếu gặp lỗi read, bỏ qua bước write
    
        # Borrow-write ngắn
        memory_persisted = True
        conn = None
        try:
            conn = await _acquire_conn(pool)
            await run_in_threadpool(persist_turn, conn, conversation_id, 
                                    payload.question, "".join(answer_parts))
            await run_in_threadpool(conn.commit)

        except Exception as exc:  # noqa: BLE001
            memory_persisted = False
            if conn is not None:
                await run_in_threadpool(conn.rollback)
            logger.error(
                "Memory persist fail for conversation %s (stream vẫn emit answer): %r", 
                conversation_id, 
                exc,
            )

        finally:
            if conn is not None:
                await run_in_threadpool(pool.putconn, conn)

        yield {
            "event": DoneEvent.event, 
            "data": DoneEvent(memory_persisted=memory_persisted).to_json(),
        }

    return EventSourceResponse(event_stream())
