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
from src.database.conversation_queries import claim_conversation, fetch_conversation_owner
from src.database.history_manager import MemoryStatus
from src.rag.qa_chain import bind_history

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
    #Borrow 1 (ngắn): ownership + eager-create, trả connection ngay,
    #không giữ conn xuyên suốt LLM call
    with pool.connection() as conn:
        owner = fetch_conversation_owner(conn, conversation_id)

        #owner khác user -> 404 chứ không 403: không lộ sự tồn tại của conversation
        if owner is not None and owner != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        if owner is None:
            claim_conversation(conn, conversation_id, user_id)
            conn.commit()

    #Borrow 2 (dài): memory đọc/ghi trong suốt lượt chain
    #build wrapper sau khi mượn conn per-request
    #memory sẽ lưu trong suốt thời gian invoke
    #sau khi xong 1 chain, trả lại conn cho pool sync
    #Hộp thư MemoryStatus mới mỗi request — dùng lại giữa các request sẽ rò
    #trạng thái turn trước vào turn sau. Handler không với tới object history
    #(RunnableWithMessageHistory gọi factory ngầm), nên cờ phải đi qua hộp thư
    status = MemoryStatus()
    with pool.connection() as conn:
        chain = bind_history(core_chain, conn, status=status)
        result = chain.invoke(
            {"question": payload.question},
            config={"configurable": {"session_id": conversation_id}},
        )

    #Cú ghi bị CallbackManager nuốt (raise_error=False), nên lỗi không tự
    #surface — log error kèm lý do là nửa còn lại của mục tiêu quan sát được.
    if status.persisted is False:
        logger.error(
            "memory persist thất bại cho conversation %s (vẫn trả 200 kèm answer): %s",
            conversation_id,
            status.error,
        )

    return ChatResponse(
        answer=result["answer"],
        sources=serialize_sources(result.get("context")),
        #persisted is True -> true; False hoặc None (chưa từng tới bước ghi)
        #-> false — đối với client, None cũng nghĩa là "chưa lưu"
        memory_persisted=status.persisted is True,
    )
