"""
    Định nghĩa lifespan của app, như mở pool connection, startup, resources ready, shutdown,...
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.database.async_connection import create_async_pool

#Decorator để tạo lifespan cho app, phase startup -> tạo kết nối async tới database (tức load các resources)
#Nếu không có decorator, lập tức khối lifespan này chỉ như 1 tác vụ thực hiện async và await thông thường mà không có phân biệt rõ các phase
@asynccontextmanager 
async def lifespan(app: FastAPI):
    pool = create_async_pool()
    await pool.open() #Tạo một pool global, sau đó với các connection đang rảnh hiện có, từng user lần lượt sẽ mượn 1 connection
                      #-> Khi thực hiện xong, sẽ trả connection đã mượn về trạng thái đang rảnh 
    app.state.db_pool = pool #Với các dòng trước khối yield, đều sẽ được thực hiện (cụ thể là mở pool)
    try:
        yield #Đây là lúc khi resources đã hoàn toàn ready, bắt đầu yield và serve user requests
              #Điểm lưu ý là, chỉ khi nào còn connection rảnh thì user mới có thể tiếp tục chat (tức có limit pool)
    finally:
        await pool.close() #Khối này để quy định rằng khi shutdown server, lập tức clean up để tránh resource leakage