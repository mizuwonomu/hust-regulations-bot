"""Main frontend entry"""

import asyncio
try:
    asyncio.get_event_loop() #Check if event loop already exists
except RuntimeError: #phòng trường hợp event tạo ở 1 main thread khác
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import sys
import os
sys.path.append(os.path.abspath('.'))
import streamlit as st

from src.rag.qa_chain import get_chain
from src.rag.embedding_utils import get_embedding_model
from src.rag.reranker_utils import load_reranker
from src.rag.config import RETRIEVER_TOP_K, LLM_TEMPERATURE

from src.database.connection import get_db_connection

from frontend.components.new_chat import render_new_chat_button
from frontend.components.sidebar import render_sidebar
from frontend.components.sidebar_title_polling import poll_sidebar_title
from frontend.components.feedback import render_feedback
from frontend.components.source_panel import render_sources
from frontend.deps import AppDeps
from frontend.services.title_generation_client import schedule_title_generation
from frontend.state.session_state import bootstrap_session_state, reset_conversation_state
from frontend.workflows.chat_flow import handle_query


#Init streamlit session state
bootstrap_session_state()

st.set_page_config(
    page_title="HUST Regulations Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

embedding_model = get_embedding_model()
reranker_model = load_reranker()
rag_chain = get_chain(k=RETRIEVER_TOP_K, temperature=LLM_TEMPERATURE, embedding_model=embedding_model, _reranker_model=reranker_model)
deps = AppDeps(
    rag_chain=rag_chain,
    db_connection_factory=get_db_connection,
    title_generation_scheduler=schedule_title_generation,
)

render_new_chat_button(reset_conversation_state)

render_sidebar(deps.db_connection_factory)
poll_sidebar_title()
    
#CHI ve 1 an duy nhat - tranh loi double display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))


#render UI
def hero_section():
    #HERO SECTION: display when no messages are found

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div style = "text-align: center;">
        <h1>🤖 HUST Regulations Bot </h1>
        <p> Trợ lý AI hỗ trợ tra cứu Quy chế đào tạo ĐHBK Hà Nội. </p>
        <p style= "color: grey; font-sizze: 0.9em;"> 👋 Chào mừng bạn đến với trợ lý AI! Dữ liệu dựa trên văn bản hợp nhất 2025
        , nếu bạn có bất kì câu hỏi nào về quy chế, hoặc đơn giản là muốn nói chuyện vui vẻ, trò chuyện,
        mình sẽ sẵn sàng hỗ trợ!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    #Tao 2 cot cho nut goi y
    col1, col2 = st.columns(2)


    suggestions = [
        "Cách tính điểm học phần",
        "Quy định về học phí",
        "Quy định về nghỉ học tạm thời",
        "Học phần song hành là gì"
    ]

    with col1:
        if st.button(suggestions[0], use_container_width=True):
            handle_query(suggestions[0], deps)
            st.rerun()
        
        if st.button(suggestions[2], use_container_width=True):
            handle_query(suggestions[2], deps)
            st.rerun()

    with col2:
        if st.button(suggestions[1], use_container_width=True):
            handle_query(suggestions[1], deps)
            st.rerun()
        
        if st.button(suggestions[3], use_container_width=True):
            handle_query(suggestions[3], deps)
            st.rerun()

if not st.session_state.messages:
    hero_section()


#input handling

#First - kiem tra trigger tu button (uu tien 1)
if "prompt_trigger" in st.session_state:
    prompt = st.session_state.prompt_trigger
    del st.session_state.prompt_trigger
    handle_query(prompt, deps)
    st.rerun()


#Roi moi kiem tra chat input UI (uu tien 2)
#placeholder bang tieng viet
elif prompt := st.chat_input("Nhập câu hỏi về quy chế, hoặc chat chit..."):
    handle_query(prompt, deps)
    st.rerun()

render_feedback()