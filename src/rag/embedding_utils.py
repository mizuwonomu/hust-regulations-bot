import streamlit as st
import torch
from langchain_huggingface import HuggingFaceEmbeddings

def get_device():
    if torch.cuda.is_available():
        return "cuda"

@st.cache_resource(show_spinner="Loading model...")
def get_embedding_model():
    device = get_device()
    print(f"Đang load embedding model 'BAAI/bg3-m3'.... {device.upper()}")

    model_name = "BAAI/bge-m3"

    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embedding_model
    