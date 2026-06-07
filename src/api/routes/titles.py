"""
    FastAPI routes cho việc sinh title.

    POST endpoint sẽ schedule background title job, còn GET endpoint 
    sẽ trả về state sẵn sàng của title cho Streamlit polling.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, status

from psycopg_pool import AsyncConnectionPool

from src.api.dependencies import get_current_user_id, get_db_pool
from src.api.schemas.titles import ConversationTitleResponse, TitleGenerationResponse
from src.database.conversation_queries_async import get_conversation_title_state
from src.services.conversation_title_service import generate_conversation_title

router = APIRouter(prefix="/conversations", tags=["titles"])


@router.post(
    "/{conversation_id}/title-generation",
    response_model=TitleGenerationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def schedule_title_generation(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    user_id: Annotated[str, Depends(get_current_user_id)],
    pool: Annotated[AsyncConnectionPool, Depends(get_db_pool)],
) -> TitleGenerationResponse:
    state = await get_conversation_title_state(pool, conversation_id, user_id)

    if state == "ready":
        return TitleGenerationResponse(
            conversation_id=conversation_id,
            status="already_ready",
        )

    if state == "missing":
        return TitleGenerationResponse(
            conversation_id=conversation_id,
            status="not_ready",
        )

    background_tasks.add_task(generate_conversation_title, pool, conversation_id, user_id)
    return TitleGenerationResponse(conversation_id=conversation_id, status="scheduled")


@router.get(
    "/{conversation_id}/title",
    response_model=ConversationTitleResponse,
)
async def get_conversation_title(
    conversation_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    pool: Annotated[AsyncConnectionPool, Depends(get_db_pool)],
) -> ConversationTitleResponse:
    state, title = await get_conversation_title_state(
        pool,
        conversation_id,
        user_id,
        include_title=True,
    )
    return ConversationTitleResponse(
        conversation_id=conversation_id,
        status=state,
        title=title,
    )
