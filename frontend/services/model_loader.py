"""
    Streamlit-side singleton cho các model nặng

    Streamlit vẫn cần singleton theo process vì mỗi lần rerun là
    chạy lại cả script — gọi thẳng hàm thuần thì model reload mỗi rerun.

    Module này là code TẠM: khi frontend chuyển hẳn sang HTTP (POST /chat),
    xóa module này và mọi import model nặng khỏi frontend
"""

from __future__ import annotations

import streamlit as st

from src.rag.config import LLM_TEMPERATURE, RETRIEVER_TOP_K
from src.rag.embedding_utils import get_embedding_model as _load_embedding_model
from src.rag.qa_chain import get_chain as _build_chain
from src.rag.reranker_utils import load_reranker as _load_reranker


@st.cache_resource(show_spinner="Đang load model...")
def get_rag_chain():
    """Build embedding + reranker + chain một lần, giữ tới khi Streamlit
    clear cache / process restart"""
    embedding_model = _load_embedding_model()
    reranker_model = _load_reranker()
    return _build_chain(
        k=RETRIEVER_TOP_K,
        temperature=LLM_TEMPERATURE,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )
