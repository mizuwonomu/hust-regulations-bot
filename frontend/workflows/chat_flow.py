"""
    Module orchestrate frontend Streamlit.

    Với từng messages nhận được từ câu hỏi của user, tạo ra một session id cho cuộc hội thoại hiện tại
    Với answer trả về từ RAG chain, sẽ stream token cùng với nguồn tài liệu truy vấn và hiển thị ra UI.

    Đồng thời, khi đã đủ lượng messages trong state, status sinh title hợp lệ, và backend verify messages type
    -> sẽ trigger sinh title và poll sync sidebar để rerun, trả về title hiển thị ra UI.
"""
import streamlit as st
from frontend.workflows.chat_stream import render_streamed_ai_answer
from src.rag.qa_chain import bind_history

def handle_query(question: str, deps):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    session_id = st.session_state.conv_id
    conn = deps.db_connection_factory()
    wrapped_chain = bind_history(deps.rag_chain, conn) #rag_chain giờ là core chain, bọc history theo từng run
    full_response, sources = render_streamed_ai_answer(wrapped_chain, question, session_id)

    #luu cau tra loi cua AI vao history de hien thi
    st.session_state.messages.append({
        "role": "ai", 
        "content": full_response,
        "sources": sources #avoid losing sources when reload
    })

    user_id = st.session_state.user_id
    message_count = len(st.session_state.messages) #khi bắt đầu có câu hỏi đầu tiên của user + answer của llm
                                                   #tức 2 message trong state -> lập tức run task sinh title

    is_first_exchange = message_count == 2
    already_scheduled = session_id in st.session_state.title_generation_started

    if is_first_exchange and not already_scheduled:
        status = deps.title_generation_scheduler(session_id, user_id)
        if status in {"scheduled", "already_ready"}:
            st.session_state.title_generation_started.add(session_id)
            st.session_state.pending_sidebar_title_sync = status == "scheduled"
