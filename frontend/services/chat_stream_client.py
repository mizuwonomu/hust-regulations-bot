"""
    Streamlit-side SSE client cho chat turn dạng token stream
"""

from __future__ import annotations

import json
import logging

import httpx
from httpx_sse import connect_sse

from frontend.services.http_config import API_BASE_URL, auth_headers, chat_timeout

logger = logging.getLogger(__name__)

#tên 4 frame type trên wire
EVENT_SOURCES = "sources"
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"


class ChatTokenStream:
    """Iterator token text + side-channel của một chat stream.

    - sources: list[dict] {title, content, doc_id} - dùng lại y nguyên cho
      render_sources như JSON path.
    - memory_persisted: default False - stream kết thúc thiếu done frame
      được coi là chưa lưu.
    - error: message nếu stream bị cắt giữa đường (error event hoặc mất kết
      nối); None nghĩa là stream lành.
    """

    def __init__(self, cm, event_source, http: httpx.Client, owns_client: bool):
        self.sources: list[dict] = []
        self.memory_persisted = False
        self.error: str | None = None
        self._cm = cm
        self._source = event_source
        self._http = http
        self._owns_client = owns_client
        self._closed = False
        self._consumed = False

    def __iter__(self):
        #stream HTTP không tua lại được - consume một lần, lần sau rỗng
        if self._consumed:
            return iter(())
        self._consumed = True
        return self._drain()

    def _drain(self):
        try:
            for sse in self._source.iter_sse():
                data = json.loads(sse.data)

                if sse.event == EVENT_TOKEN:
                    yield data["text"]

                elif sse.event == EVENT_SOURCES:
                    #capture-once ở phía route; giá trị cuối là giá trị đúng
                    self.sources = data["sources"]

                elif sse.event == EVENT_DONE:
                    self.memory_persisted = bool(data["memory_persisted"])
                    return
                
                elif sse.event == EVENT_ERROR:
                    self.error = data["message"]
                    return
        
        except httpx.HTTPError:
            #mất kết nối giữa stream - giữ token đã yield, báo cờ cho UI
            logger.exception("SSE stream đứt giữa đường.")
            self.error = "Mất kết nối với máy chủ giữa stream."
            return
        finally:
            self._close()

    def _close(self) -> None:
        #đóng đúng một lần: iteration kết thúc bình thường, break giữa đường
        #hay consumer abandon generator (GeneratorExit) đều chạy qua đây
        if not self._closed:
            self._closed = True
            self._cm.__exit__(None, None, None)
            if self._owns_client:
                self._http.close()


def stream_message(
    conversation_id: str,
    user_id: str,
    question: str,
    *,
    client: httpx.Client | None = None,
) -> ChatTokenStream | None:
    """Mở SSE stream cho một chat turn; ChatTokenStream nếu 200, None nếu
    request fail (404 / non-200 / transport).

    Stream được mở eager tại đây (headers đã về) để 404/non-200 quyết định
    no-answer trước khi caller bắt đầu render. Seam `client=` cho test: bỏ qua
    thì hàm tự sở hữu + đóng httpx.Client (timeout chat dài); truyền vào thì
    mượn, KHÔNG đóng client của caller - đóng xảy ra khi iteration kết thúc.
    """
    owns_client = client is None
    http = client if client is not None else httpx.Client(timeout=chat_timeout())
    try:
        cm = connect_sse(
            http,
            "POST",
            f"{API_BASE_URL}/conversations/{conversation_id}/messages/stream",
            headers=auth_headers(user_id),
            json={"question": question},
        )
        event_source = cm.__enter__()

        status = event_source.response.status_code
        if status == 404:
            logger.warning(
                "Conversation %s trả 404 cho user %s - không tồn tại hoặc "
                "không phải của user này",
                conversation_id,
                user_id,
            )
            cm.__exit__(None, None, None)
            if owns_client:
                http.close()
            return None
        if status != 200:
            #500 từ borrow-1 (pre-stream) rơi vào đây - no-answer như transport
            logger.error("Stream tới conversation %s trả %s", conversation_id, status)
            cm.__exit__(None, None, None)
            if owns_client:
                http.close()
            return None

        return ChatTokenStream(cm, event_source, http, owns_client)
    except httpx.HTTPError:
        logger.exception("Failed to open stream to conversation %s", conversation_id)
        if owns_client:
            http.close()
        return None
