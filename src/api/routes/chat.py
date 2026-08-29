"""
    FastAPI route cho một lượt chat.

    Handler là def (không phải async def): core chain là sync và block,
    nên để Starlette đẩy nó ra threadpool thay vì chạy trên event loop
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg_pool import ConnectionPool

from src.api.dependencies import get_current_user_id, get_rag_chain, get_sync_db_pool
from src.api.schemas.chat import ChatRequest, ChatResponse, serialize_sources
from src.database.conversation_queries import (
    claim_conversation,
    fetch_conversation_owner,
)
from src.database.history_manager import persist_turn, read_history

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["chat"])


@router.post("/{conversation_id}/messages", response_model=ChatResponse)
def post_message(
    conversation_id: str,
    payload: ChatRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    pool: Annotated[ConnectionPool, Depends(get_sync_db_pool)],
    core_chain: Annotated[object, Depends(get_rag_chain)],
) -> ChatResponse:
    #Borrow 1 (ngắn): ownership + eager-create + read history, trả connection ngay,
    #không giữ conn xuyên suốt LLM call
    with pool.connection() as conn:
        owner = fetch_conversation_owner(conn, conversation_id)

        #owner khác user -> 404 chứ không 403: không lộ sự tồn tại của conversation
        if owner is not None and owner != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        if owner is None:
            claim_conversation(conn, conversation_id, user_id)

        conversation_history = read_history(conn, conversation_id)

    # Không giữ conn trong suốt thời gian invoke
    result = core_chain.invoke(
        {"question": payload.question, "chat_history": conversation_history},
        config={"configurable": {"session_id": conversation_id}},
    )

    memory_persisted = True # Thành công lưu messages vào DB
    try:
        with pool.connection() as conn: # Borrow-write history: with tự commit khi thoát chain
            persist_turn(conn, conversation_id, payload.question, result["answer"])

    except Exception as exc: # noqa: BLE001
        memory_persisted = False
        logger.error("Memory persist fail for %s: %r", conversation_id, exc)

    return ChatResponse(
        answer=result["answer"],
        sources=serialize_sources(result.get("context")),
        memory_persisted=memory_persisted,
    )
