"""
    Module orchestrate frontend Streamlit.

    Khi đủ lượng messages trong state và là first exchange -> trigger sinh title
    và poll sync sidebar để rerun, trả về title hiển thị ra UI
"""
import streamlit as st

from frontend.services.chat_stream_client import stream_message
from frontend.workflows.chat_stream import (
    NO_ANSWER_MESSAGE,
    classify_turn_outcome,
    render_streamed_ai_answer,
)


def handle_query(question: str, deps):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    session_id = st.session_state.conv_id
    user_id = st.session_state.user_id

    stream = stream_message(session_id, user_id, question)

    if stream is None:
        # Hard fail (404 / transport / non-200): không append AI message và
        # không trigger title -> để user hỏi lại. User message đã append thì
        # giữ nguyên
        with st.chat_message("ai"):
            st.error(NO_ANSWER_MESSAGE)
        return

    full_response, sources = render_streamed_ai_answer(stream)

    if classify_turn_outcome(stream, full_response) != "complete":
        # Incomplete turn: no_answer (error trước token, đã st.error) hay partial
        # (error sau token, đã render + st.error) đều không append AI message và
        # không trigger title - incomplete turn không được sống sót qua reload
        return

    # Lưu câu trả lời của AI vào state để hiển thị
    st.session_state.messages.append({
        "role": "ai",
        "content": full_response,
        "sources": sources,  # avoid losing sources when reload
    })

    message_count = len(st.session_state.messages)  # 2 = first exchange (user + AI)
    is_first_exchange = message_count == 2
    already_scheduled = session_id in st.session_state.title_generation_started

    if is_first_exchange and not already_scheduled:
        status = deps.title_generation_scheduler(session_id, user_id)
        if status in {"scheduled", "already_ready"}:
            st.session_state.title_generation_started.add(session_id)
            st.session_state.pending_sidebar_title_sync = status == "scheduled"
