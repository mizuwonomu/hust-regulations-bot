LLM_TEMPERATURE = 0.1
RETRIEVER_TOP_K = 15

#Tỉ lệ tương đối so với top-1 (score cross-encoder KHÔNG so sánh được giữa các query)
#nếu score reranker < RERANK_RATIO * top-1 thì sẽ không xét đến
RERANK_RATIO = 0.45
RERANK_MAX_CHILDREN = 5 #số lượng rerank docs tối đa để xét

#Model identifiers
ROUTER_MODEL = "qwen/qwen3.6-27b"
QUERY_REWRITE_MODEL = "qwen/qwen3.6-27b"
CHITCHAT_MODEL = "qwen/qwen3.6-27b"
INFERENCE_MODEL = "openai/gpt-oss-120b"
TITLE_GENERATOR_MODEL = "qwen/qwen3.6-27b"
JUDGE_MODEL = "qwen/qwen3.6-27b"

#Temperatures
ROUTER_TEMPERATURE = 0.0
QUERY_REWRITE_TEMPERATURE = 0.2
CHITCHAT_TEMPERATURE = 0.7

#Vector store / doc store path collection
CHROMA_PATH = "chroma_db"
CHROMA_COLLECTION = "split_parents"
DOC_STORE_PATH = "doc_store_pdr"
