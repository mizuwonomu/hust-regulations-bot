"""
    Định nghĩa lifespan của app: load model/chain một lần, mở DB pool,
    dọn dẹp khi shutdown
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.database.async_connection import create_async_pool
from src.database.sync_connection import create_sync_pool
from src.rag.config import LLM_TEMPERATURE, RETRIEVER_TOP_K
from src.rag.embedding_utils import get_embedding_model
from src.rag.qa_chain import get_chain
from src.rag.reranker_utils import load_reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model = get_embedding_model()
    reranker_model = load_reranker()
    rag_chain = get_chain(
        k=RETRIEVER_TOP_K,
        temperature=LLM_TEMPERATURE,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )

    app.state.embedding_model = embedding_model
    app.state.reranker_model = reranker_model
    app.state.rag_chain = rag_chain

    
    async_pool = create_async_pool() # Async pool: title + SQL đọc
    sync_pool = create_sync_pool() # Sync pool: dành cho mỗi memory path runnable

    await async_pool.open() #Tạo một pool global, sau đó với các connection đang rảnh hiện có, từng user lần lượt sẽ mượn 1 connection
                      #-> Khi thực hiện xong, sẽ trả connection đã mượn về trạng thái đang rảnh 
    
    sync_pool.open()  # sync pool mở đồng bộ (dùng trong threadpool lúc chat)

    app.state.db_pool = async_pool #Với các dòng trước khối yield, đều sẽ được thực hiện (cụ thể là mở pool)
    app.state.sync_db_pool = sync_pool

    try:
        yield #Đây là lúc khi resources đã hoàn toàn ready, bắt đầu yield và serve user requests
              #Điểm lưu ý là, chỉ khi nào còn connection rảnh thì user mới có thể tiếp tục chat (tức có limit pool)
    finally:
        await async_pool.close() #Khối này để quy định rằng khi shutdown server, lập tức clean up để tránh resource leakage
        sync_pool.close()
