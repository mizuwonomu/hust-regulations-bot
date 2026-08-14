LLM_TEMPERATURE = 0.1
RETRIEVER_TOP_K = 15

#Tỉ lệ tương đối so với top-1 (score cross-encoder KHÔNG so sánh được giữa các query)
#nếu score reranker < RERANK_RATIO * top-1 thì sẽ không xét đến
RERANK_RATIO = 0.45
RERANK_MAX_CHILDREN = 5 #số lượng rerank docs tối đa để xét

#Model identifiers
ROUTER_MODEL = "llama-3.1-8b-instant"
QUERY_REWRITE_MODEL = "llama-3.3-70b-versatile"
CHITCHAT_MODEL = "llama-3.3-70b-versatile"
INFERENCE_MODEL = "openai/gpt-oss-120b"

#Temperatures
ROUTER_TEMPERATURE = 0.0
QUERY_REWRITE_TEMPERATURE = 0.2
CHITCHAT_TEMPERATURE = 0.7

#Vector store / doc store path collection
CHROMA_PATH = "chroma_db"
CHROMA_COLLECTION = "split_parents"
DOC_STORE_PATH = "doc_store_pdr"
