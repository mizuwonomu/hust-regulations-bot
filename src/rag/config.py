LLM_TEMPERATURE = 0.1
RETRIEVER_TOP_K = 15

#Tỉ lệ tương đối so với top-1 (score cross-encoder KHÔNG so sánh được giữa các query)
#nếu score reranker < RERANK_RATIO * top-1 thì sẽ không xét đến 
RERANK_RATIO = 0.6
RERANK_MAX_CHILDREN = 5 #số lượng rerank docs tối đa để xét