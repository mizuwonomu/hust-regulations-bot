from __future__ import annotations

from fastapi import FastAPI

from src.api.lifespan import lifespan
from src.api.routes.chat import router as chat_router
from src.api.routes.titles import router as titles_router

app = FastAPI(title="HUST RAG API", lifespan=lifespan)
app.include_router(titles_router)
app.include_router(chat_router)

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "OK"}