"""
    Module render chat turn dạng token stream.

    stream_handler consume ChatTokenStream từ frontend/services/chat_stream_client,
    yield token text cho st.write_stream; sau khi drain ghi sources vào
    st.session_state.last_context (mirror pattern full_context của bản
    chain.stream cũ). render_streamed_ai_answer bọc write_stream, render sources
    và hiển thị warning/error theo phân loại; chat_flow đọc classify_turn_outcome
    để quyết định append state và trigger title
"""

import streamlit as st

from frontend.components.source_panel import render_sources
from frontend.services.chat_stream_client import ChatTokenStream

# Thông điệp hard-fail dùng chung cho no-answer signal (client None) và error
# Event trước token nào - một chuỗi, hai nơi hiển thị, không drift
NO_ANSWER_MESSAGE = "Không nhận được câu trả lời từ máy chủ. Vui lòng thử lại."

NOT_PERSISTED_MESSAGE = "Câu trả lời đã hiển thị nhưng không được lưu vào lịch sử hội thoại."


def stream_handler(stream: ChatTokenStream):
    """Yield token text cho st.write_stream; sau khi drain ghi sources vào
    session_state. write_stream block tới khi drain,
    còn sources đã nằm sẵn trong stream object từ giữa iteration"""
    for token in stream:  # noqa: UP028
        yield token
    st.session_state.last_context = stream.sources


def classify_turn_outcome(stream, full_response: str) -> str:
    """Phân loại turn theo biên hard/soft khớp JSON path (G6):
    - "complete": stream lành - kể cả memory_persisted=False (chỉ soft warning).
    - "partial": error event SAU khi đã có token - incomplete turn, KHÔNG append
      vào state, KHÔNG trigger title.
    - "no_answer": error event TRƯỚC token nào - hard fail, tương đương None.
    """
    if stream.error is not None:
        return "partial" if full_response else "no_answer"
    return "complete"


def render_streamed_ai_answer(stream: ChatTokenStream) -> tuple[str, list[dict]]:
    """Wrap st.write_stream + render sources + hiển thị warning/error theo
    outcome; trả (full_response, sources) cho chat_flow quyết định append state
    và title."""
    with st.chat_message("ai"):
        full_response = st.write_stream(stream_handler(stream))
        sources = st.session_state.get("last_context", [])
        render_sources(sources)

        outcome = classify_turn_outcome(stream, full_response)
        if outcome == "no_answer":
            st.error(NO_ANSWER_MESSAGE)
        elif outcome == "partial":
            st.error(f"Câu trả lời bị cắt giữa đường: {stream.error}")
        elif not stream.memory_persisted:
            # Soft degrade: answer đã sinh thành công server-side, chỉ phần ghi
            # history thất bại -> cảnh báo, không vứt bỏ lượt chat này
            st.warning(NOT_PERSISTED_MESSAGE)

    return full_response, sources
