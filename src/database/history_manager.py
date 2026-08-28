import asyncio
from dataclasses import dataclass

from langchain_postgres import PostgresChatMessageHistory


@dataclass
class MemoryStatus:
    """Hộp thư ghi kết quả persist memory — do caller (chat handler) sở hữu.

    RunnableWithMessageHistory gọi history factory ngầm bên trong pipeline,
    handler không với tới object history, nên tín hiệu phải đi qua một hộp
    thư đưa sẵn xuống cho tầng dưới ghi vào.

    persisted:
        True  — cú ghi Postgres đã hoàn tất
        False — cú ghi ném exception (lý do nằm ở error)
        None  — chưa từng chạy tới bước ghi (lỗi xảy ra trước đó, vd đường đọc)

    error: lý do fail (repr của exception) để log
    """

    persisted: bool | None = None
    error: str | None = None


class TrackedPostgresHistory(PostgresChatMessageHistory):
    """PostgresChatMessageHistory + ghi cờ kết quả cú ghi vào hộp thư.

    Override add_messages để ghi cờ; aget_messages/aadd_messages delegate về
    bản sync qua asyncio.to_thread - astream của RunnableWithMessageHistory
    (route SSE) đi qua đường đọc/ghi async mà langchain_postgres hard-reject
    khi history dựng từ sync conn.

    Vẫn raise exception: giữ đúng ngữ nghĩa và log của LangChain vẫn có;
    cờ trong hộp thư mới là thứ thật sự mang tín hiệu ra ngoài (exception
    của cú ghi bị CallbackManager nuốt ở tầng listener).

    status=None (mặc định): bỏ qua việc ghi cờ.
    """

    def __init__(
        self,
        table_name: str,
        session_id: str,
        /,
        *,
        status: MemoryStatus | None = None,
        **kwargs,
    ):
        #**kwargs chứ không kê cứng sync_connection: giữ nguyên chữ ký của cha
        #(cha nhận cả sync_connection lẫn async_connection). Kê cứng sẽ thu hẹp
        #class con và làm vỡ mọi caller dùng async_connection sau này.
        super().__init__(table_name, session_id, **kwargs)
        self._status = status

    def add_messages(self, messages) -> None:
        try:
            super().add_messages(messages)
            if self._status is not None:
                self._status.persisted = True
        except Exception as exc:
            if self._status is not None:
                self._status.persisted = False
                self._status.error = repr(exc) #repr giữ lại loại exception, ví dụ như OperationalError
            raise

    async def aget_messages(self) -> list:
        # astream của wrapper đọc history qua đường async - delegate về bản
        # sync chạy trong executor: conn sync vẫn được dùng tuần tự
        return await asyncio.to_thread(self.get_messages)

    async def aadd_messages(self, messages) -> None:
        # Ghi trên đường async cũng phải đi qua add_messages của chính class
        # này - hộp thư MemoryStatus ghi cờ đúng trên cả astream, không chỉ invoke
        await asyncio.to_thread(self.add_messages, messages)


def get_postgres_history(
    conn, conversation_id: str, status: MemoryStatus | None = None
) -> TrackedPostgresHistory:
    #sử dụng thẳng conversation_id uuid4 để map với bảng conversations
    #với user_id
    #create history object
    history = TrackedPostgresHistory(
        "chat_history",
        conversation_id,
        sync_connection=conn,
        status=status,
    )

    return history
