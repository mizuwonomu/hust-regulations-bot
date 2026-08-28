"""
    FastAPI routes cho read path của conversation: list + load old history's messages
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from psycopg_pool import AsyncConnectionPool

from src.api.dependencies import get_current_user_id, get_db_pool
from src.api.schemas.conversations import (
    ConversationListResponse,
    ConversationMessage,
    ConversationMessagesResponse,
    ConversationSummary,
)
from src.database.conversation_queries_async import (
    fetch_conversation_messages,
    fetch_conversation_owner_and_title_pool,
    list_user_conversations,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    user_id: Annotated[str, Depends(get_current_user_id)],
    pool: Annotated[AsyncConnectionPool, Depends(get_db_pool)],
) -> ConversationListResponse:
    rows = await list_user_conversations(pool, user_id)
    return ConversationListResponse(
        conversations=[
            ConversationSummary(conversation_id=conv_id, title=title)
            for conv_id, title in rows
        ]
    )


@router.get("/{conversation_id}/messages", response_model=ConversationMessagesResponse)
async def get_conversation_messages(
    conversation_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    pool: Annotated[AsyncConnectionPool, Depends(get_db_pool)],
) -> ConversationMessagesResponse:
    conversation = await fetch_conversation_owner_and_title_pool(pool, conversation_id) # Lookup owner trước
    if conversation is None or conversation[0] != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) # 404: trả về không tồn tại <-> 403: forbidden, có tồn tại nhưng không đủ authorization

    messages = await fetch_conversation_messages(pool, conversation_id, user_id) # Sau đó, load messages của user, conversation id đó
    return ConversationMessagesResponse(
        conversation_id=conversation_id,
        messages=[
            ConversationMessage(role=message["role"], content=message["content"])
            for message in messages
        ],
    ) # Trả về toàn bộ message của conversation id đó