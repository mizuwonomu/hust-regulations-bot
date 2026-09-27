"""Harness so sánh router production (Groq) với Jev trên corpus định tuyến 30 câu.

Package này cô lập hoàn toàn: chỉ gọi bước định tuyến, không retrieval, không sinh
answer, không database, và không import src/rag/qa_chain.py vì module đó gọi
load_dotenv lúc import
"""
