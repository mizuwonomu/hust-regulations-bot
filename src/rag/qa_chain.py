import os
import sys
sys.path.append(os.path.abspath('.'))
import pickle
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch
#parallel: chạy nhiều nhánh xử lý cùng 1 lúc, lambda: định nghĩa lambda nhưng thiết kế theo 
#dạng trigger on time. Passthrough: truyền type on time
from src.database.history_manager import get_postgres_history
from src.database.connection import get_db_connection
from src.rag.config import LLM_TEMPERATURE, RETRIEVER_TOP_K, RERANK_RATIO, RERANK_MAX_CHILDREN

from langsmith import traceable

from dotenv import load_dotenv
load_dotenv()


CHROMA_PATH = "chroma_db"
DOC_STORE_PATH = "doc_store_pdr"

def format_docs(docs):
    formatted = []
    #Van la ep string tu document object, nhung co them cac metadata de danh dau
    
    for i, doc in enumerate(docs):
        source_title = doc.metadata.get("title", f"Điều khoản {i+1}")
        content = doc.page_content.replace("\n", " ")
        formatted.append(f"Source [{i+1}] ({source_title}):\n{content}")

    return "\n\n".join(formatted)

#connect database
def get_session_history(session_id: str):
    conn = get_db_connection()

    return get_postgres_history(conn, session_id)


#base model pydantic output for query rephrasing (multi-query expansion)
#Aka query decomposition
class QueryExpansion(BaseModel):
    reasoning: str = Field(description="Phân tích ngắn gọn ý định của câu hỏi gốc") #Field: chú thích rõ key để làm gì
    queries: List[str] = Field(description="Danh sách tối đa 3 câu hỏi đơn lẻ bằng tiếng Việt để tìm kiếm")

@traceable(run_type='chain')
@st.cache_resource
def get_chain(k, temperature, embedding_model, _reranker_model):
    embedding_model = embedding_model
    #load vector store
    vector_store = Chroma(
        collection_name= "split_parents",
        embedding_function = embedding_model,
        persist_directory = CHROMA_PATH
    )

    fs = LocalFileStore(DOC_STORE_PATH)
    doc_store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )

    child_vector_retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {'k': k}
    )

    child_data = vector_store.get()

    from langchain_core.documents import Document

    all_child_docs = [
        Document(page_content=txt, metadata=md)
        for txt, md in zip(child_data['documents'], child_data['metadatas'])
    ]

    bm25_retriever = BM25Retriever.from_documents(all_child_docs)
    bm25_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers = [child_vector_retriever, bm25_retriever],
        weights = [0.5, 0.5]
    )

    #hugging-face based reranker (bge-reranker-v2-m3 by default)
    reranker = _reranker_model

    # ── Traceable helpers (child spans visible in LangSmith) ──────────────────
    # Đóng gói 3 bước "trần" bên trong retreive_parents thành span riêng.
    # Heavy objects (reranker, doc_store) ở closure — không lòi vào trace input.

    @traceable(run_type="tool", name="rerank_children")
    def _rerank_children(original_question: str, candidates: list) -> list:
        """Chấm điểm toàn bộ candidate children, trả về list đã sort desc."""
        pairs = [(original_question, doc.page_content) for doc in candidates]
        scores = reranker.predict(pairs)

        scored = sorted(
            zip(candidates, [float(s) for s in scores]),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for rank, (doc, score) in enumerate(scored):
            doc.metadata["rerank_score"] = score
            result.append({
                "rank": rank,
                "title": doc.metadata.get("title", ""),
                "doc_id": doc.metadata.get("doc_id", ""),
                "score": score,
                "doc": doc,
            })
        return result

    @traceable(run_type="tool", name="apply_score_ratio")
    def _apply_score_ratio(scored_list: list, ratio: float) -> dict:
        """Chọn child theo TỈ LỆ tương đối với top-1, không dùng sàn tuyệt đối

        - top-1 luôn được giữ (nó là ứng viên tốt nhất mà retrieval có)
        - rank>=2 chỉ được giữ nếu score >= top_score * ratio
        """

        if not scored_list:
            return {
                "n_candidates": 0,
                "n_kept": 0,
                "n_dropped": 0,
                "max_score": 0.0,
                "ratio": ratio,
                "cutoff": 0.0,
                "kept": [],
                "docs": [],
            }

        top_score = scored_list[0]["score"]
        cutoff = top_score * ratio

        #top-1 vô điều kiện, phần còn lại phải bám đủ sát top-1
        keep_items = [scored_list[0]]
        keep_items.extend(item for item in scored_list[1:] if item["score"] >= cutoff)

        kept_items = keep_items[:RERANK_MAX_CHILDREN]
        return {
            "n_candidates": len(scored_list),
            "n_kept": len(kept_items),
            "n_dropped": len(scored_list) - len(kept_items),
            "max_score": top_score,
            "ratio": ratio,
            "cutoff": cutoff,
            "kept": [
                {"title": i["title"], "doc_id": i["doc_id"], "score": i["score"]}
                for i in kept_items
            ],
            "docs": [i["doc"] for i in kept_items],
        }

    @traceable(run_type="tool", name="fetch_parents")
    def _fetch_parents(requested_parent_ids: list) -> dict:
        """Lấy parent docs từ doc_store, lộ rõ missing keys (mget trả None)."""
        raw = doc_store.mget(requested_parent_ids)
        found, missing = [], []
        for pid, doc in zip(requested_parent_ids, raw):
            if doc is not None:
                found.append(pid)
            else:
                missing.append(pid)
        parents = [doc for doc in raw if doc is not None]
        return {
            "requested_ids": requested_parent_ids,
            "found_ids": found,
            "missing_ids": missing,
            "n_parents_returned": len(parents),
            "docs": parents,
        }

    #Custom chain de lay parent:
    #Query -> Ensemble -> List[child] -> rerank -> extract ids -> docstore -> list[parent]
    def retreive_parents(input_data):
        original_question = input_data["input_pass"]["question"] #lấy lại câu hỏi gốc
        #find child
        query_obj = input_data["standalone_question"] #object pydantic (str and list str)

        sub_queries = query_obj.queries #lấy attribute list "queries" từ class pydantic
        if not sub_queries:
            sub_queries = [original_question] #backup if llama can't make pydantic agreement

        #chạy song song 3 queries
        #kết quả là list[list[Document]] (Runnable[list[RetrieverInput], list[RetrieverOutput]])
        nested_docs = ensemble_retriever.map().invoke(sub_queries)

        #flatten list of list to a single list và lọc trùng bằng page_content
        unique_docs_dict = {}
        for sublist in nested_docs:
            if sublist: #đảm bảo không rỗng
                for doc in sublist:
                    #dùng content làm key để lọc trùng document cho từng queries invoke
                    if doc.page_content not in unique_docs_dict:
                        unique_docs_dict[doc.page_content] = doc

        candidate_child_docs = list(unique_docs_dict.values())

        if not candidate_child_docs:
            return []

        # Bước 1: rerank toàn bộ candidates (span: rerank_children)
        #sub-queries chỉ làm nhiệm vụ tăng recall, mà reranker cần precision -> phải luôn đối chiếu với câu hỏi gốc với chunks
        scored_list = _rerank_children(original_question, candidate_child_docs)

        # Bước 2: chọn theo tỉ lệ tương đối với top-1 (span: apply_score_ratio)
        ratio_result = _apply_score_ratio(scored_list, ratio=RERANK_RATIO)
        thresholded_children = ratio_result["docs"]

        if not thresholded_children:
            return []

        seen_parent_ids = set()
        unique_parent_ids = []

        for doc in thresholded_children:
            p_id = doc.metadata.get("doc_id")

            if p_id is not None and p_id not in seen_parent_ids:
                seen_parent_ids.add(p_id)
                unique_parent_ids.append(p_id)

        if not unique_parent_ids:
            return []

        # Bước 3: fetch parents (span: fetch_parents)
        fetch_result = _fetch_parents(unique_parent_ids)
        parents = fetch_result["docs"]

        max_parents = 4
        if len(parents) > max_parents:
            parents = parents[:max_parents]

        return parents
    rephrase_system_prompt = """You are a Query Transformation Engine for a Vietnamese university regulation QA system.
    Your ONLY Task: Given a chat history and a new user question, rewrite the question into a single standalone question in Vietnamese.

    Rules:
    - Output ONLY the rewritten question. Nothing else.
    - DO NOT answer human's question.
    - NEVER ask for clarification.
    - If no rewrite needed, return the original question EXACTLY as-is.
    - Preserve ALL Vietnamese legal/academic terms unchanged.
    - Generate maximum 3 sub-queries.

    {format_instructions}
    Examples:
    [No history] Query: "Quy định về học phí" -> Quy định về học phí
    [History: Quy định về học phí] Query: "Thế còn miễn giảm?" -> Quy định miễn giảm học phí tại HUST là gì?"
    """

    parser = PydanticOutputParser(pydantic_object=QueryExpansion)

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", rephrase_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]).partial(format_instructions=parser.get_format_instructions())
    #partial: luôn nhét format_instructions ngay từ lúc prompt được khởi tạo

    #ở bước rewrite, giữ nguyên từ khoá luật từ history và new input nhưng văn phong cần tự nhiên theo Việt, tránh rập khuôn -> temp 0.3
    query_rewrite_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        max_retries=0,
        temperature=0.2
    )

    #chain nhỏ chỉ làm nhiệm vụ: Input (History + Query) -> Output (List string query mới) + Plan thoughts
    rephrase_chain = rephrase_prompt | query_rewrite_llm | parser

    #Phân loại câu hỏi thuộc rag hay xã giao
    router_template = """
    M là một chuyên gia phân loại câu hỏi. Hãy đọc câu hỏi của người dùng và quyết định câu hỏi đó thuộc loại nào"

    1. 'chat': Các câu chào hỏi xã giao, hỏi thăm sức khỏe, không liên quan đến thông tin cụ thể trong tài liệu 
    - Ví dụ: 
    + Human: Hôm nay nên ăn gì nhỉ? -> Tao nghĩ hôm nay mày nên ăn phở đó! 
    + Human: Tự nhiên buồn ghê -> Sao thế, có chuyện gì m muốn nói không?
    
    2. 'RAG': 
        - Các câu hỏi liên quan đến quy chế đào tạo, Luật, Quy định nhà trường
        - Từ khóa nhận diện: Ví dụ: "học phí", "tín chỉ", "cảnh báo học tập", "điểm học phần",...
        - Các câu hỏi về thủ tục, điều kiện, thời gian học tập tại HUST.

    Chỉ được trả kết quả theo 1 từ duy nhất: 'chat' hoặc 'RAG'. KHÔNG trả lời thêm bất cứ điều gì khác.

    Câu hỏi: {question}
    Phân loại:
    """


    router_prompt = ChatPromptTemplate.from_template(router_template)

    #router bắt buộc không chứa bất cứ thông tin cảm xúc hay sáng tạo, chỉ dùng để định tuyến
    router_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        max_retries=0,
        temperature= 0.0
    )

    router_chain = router_prompt | router_llm | StrOutputParser()
    
    #Nhánh A: Chat thông thường(dùng cách xã giao của model)
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly and polite AI assistant for students at Hanoi University of Science and Technology (HUST). 
        The user is just chatting normally (not asking about academic regulations).

        Rules:
        1. ALWAYS respond in natural, conversational Vietnamese (Tiếng Việt). NEVER use English.
        2. Use a friendly tone, can use 1-2 any emojis that suits the context tone, or 1-2 Vietnamese emojis (e.g. "=))))", ":)))" )
        3. Use pronouns "t" (stands for "tao" in Vietnamese) for yourself, call user as "m" (stands for "mày" in Vietnamese).

        Example response: "Chào m nha! 😎 Tao là trợ lý AI của HUST đây. Hôm nay m có cần t hỗ trợ tra cứu quy chế đào tạo hay điểm số gì không nào?"
        + Example 2: "T nghĩ hôm nay m nên đi ăn cơm tấm đó"
        + Example 3: T có gợi ý về trò chơi rất hay nè"
        """),

        MessagesPlaceholder(variable_name="chat_history"),
        
        ("human", "{question}"),
    ])

    #do chat thông thường không nhất thiết cần dùng luật -> chọn model có khả năng nói tự nhiên, response nhanh
    chitchat_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        max_retries=0,
        temperature=0.7
    )
    
    #Chuẩn hóa output format giống rag chain, trả về dạng 辞書型
    chat_chain = (
        chat_prompt 
        | chitchat_llm 
        | StrOutputParser()
        | RunnableLambda(lambda x: {"answer": x, "context": []}) #fake content rỗng
    )

    #define prompt with role-based messages, only rely on documents
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system", """M là trợ lý AI chuyên hỗ trợ sinh viên HUST - Đại học Bách Khoa tra cứu Quy chế đào tạo (Academic Regulations)
            Nhiệm vụ của m là trả lời chính xác dựa trên Context được cung cấp.
            
            Quy tắc tuyệt đối (絶対ルール):
            1. Luôn trích dẫn rõ ràng thông tin nằm ở **Điều nào** (Dựa vào dòng đầu tiên của context). Với các con số (tín chỉ, học phí, mức cảnh báo), phải tuyệt đối chính xác.
            2. KHÔNG được sử dụng kiến thức bên ngoài (Outside knowledge) hoặc kiến thức có sẵn trong model (Pre-trained knowledge) để trả lời.
            3. Nếu thông tin không có trong Context, hãy trả lời: "Thông tin này không có trong quy chế hiện tại."
            4. KHÔNG được bịa đặt nội dung (Hallucination). 
            5. Giọng văn nghiêm túc nhưng dễ hiểu, phù hợp với sinh viên.

            Quy tắc trích dẫn (Citation Rules): 
            1. Khi đưa ra bất kỳ thông tin nào, PHẢI trích dẫn nguồn gốc theo điều, khoản nào và bằng cú pháp [index]. 
            Ví dụ:
            - "User": "Quy định về học phí"
            - Answer: "Theo quy định về học phí tại điều 9 [1], sinh viên phải..." 

            - "User": "Học phần song hành là gì?"
            - Answer: "Theo điểm c khoản 6 điều 4 [1], học phần song hành là: Học phần A là học phần song hành của học phần B thì sinh
            viên phải theo học trước hoặc học đồng thời với học phần B..."
            
            2. Nếu thông tin đến từ nhiều nguồn, hãy liệt kê đủ: [1], [2], [3],.... 
            Ví dụ:
            - "User": "Tôi có điểm CPA toàn khoá là 3,4 và tổng số tín chỉ học lại chiếm 6% tổng số tín chỉ dùng để tính điểm. Vậy tôi đang được xếp loại học lực gì và khi tốt nghiệp bằng cử nhân của tôi sẽ nhận được hạng loại gì?
            - Answer: "Với điểm CPA toàn khóa là 3,4 (nằm trong khoảng 3,2 đến 3,59), bạn sẽ được xếp loại học lực Giỏi theo khoản 6 Điều 12 [1]. Tuy nhiên, do khối lượng học phần phải học lại của bạn chiếm 6% (vượt quá mức 5% của tổng số tín chỉ được dùng tính điểm trung bình toàn khóa), hạng tốt nghiệp của bạn sẽ bị giảm đi một mức xuống thành loại Khá theo điểm a khoản 2 Điều 15 [2]."
            
            3. Cuối câu trả lời KHÔNG cần tạo danh sách tài liệu tham khảo (vì giao diện sẽ tự hiển thị).   

            Context:
            {context}
            """),
    
        MessagesPlaceholder(variable_name="chat_history"),

        ("human", "{question}"),
    ])
    
    #test hybrid with no filter
    inference_llm = ChatGroq(
        model = "openai/gpt-oss-120b",
        temperature = temperature,
        reasoning_format = "parsed"
    )

    answer_chain = (
        qa_prompt
        | inference_llm
        | StrOutputParser()
    )

    rag_chain=RunnableParallel(
        #Nhánh 1: tính toán câu hỏi độc lập

        input_pass = RunnablePassthrough(), #giữ input gốc để lấy chat_history
        #passthrough truyền nguyên 1 dict {'question', 'chat_history}

        standalone_question = rephrase_chain, #dùng llm để viết lại câu hỏi

    ).assign(
        #nhánh 2: dùng câu hỏi độc lập để tìm tài liệu (context)

        context = RunnableLambda(retreive_parents)
    
        #x lúc này đã có key 'standalone_question'
        #lambda để chuyển đổi object retreive thành 1 Runnable
        #để có thể delay cho đến khi được invoke đúng thời điểm, thay vì chạy ngay tức khắc

    ).assign( #Lấy context là danh sách các chunk gốc và question
        #nhánh 3: Trả lời
        answer = RunnableBranch(
            (
                RunnableLambda(lambda x: len(x["context"]) == 0), #x: chứa dictionary của output runnableparallel 
                RunnableLambda(lambda x: "Thông tin này không có trong quy chế hiện tại.") #short-circuit: Nếu context rỗng, trả về answer no context tất định
            ),
            RunnableLambda(lambda x: { #chia nhánh runnable: chỉ gen answer khi có context
                "context": format_docs(x["context"]),
                "question": x["input_pass"]["question"],
                "chat_history": x["input_pass"]["chat_history"]
            }) #ép danh sách chunk thành string
            | answer_chain
        )
        #Đảm bảo app.py nhận 1 cục dictionary đầy đủ
    )


    def route_decision(info):
        #info: Dict chứa đầu ra của các bước trước

        #router_chain cần input là dict {'question':...}
        decision = router_chain.invoke({"question": info["question"]})

        #clean string (xóa khoảng trắng thừa nếu có)
        decision = decision.strip().lower()

        if "chat" in decision:
            return chat_chain
        else:
            return rag_chain
    
    # Tổng hợp final chain
    full_chain = RunnableLambda(route_decision)

    chain_with_history = RunnableWithMessageHistory(
        full_chain,
        get_session_history,
        input_messages_key= "question",
        history_messages_key= "chat_history",
        output_messages_key= "answer",
    )
    
    return chain_with_history

def debug_memory(session_id):

    history_obj = get_postgres_history(session_id)

    #lúc này langchain sẽ chạy câu select trong DB
    messages = history_obj.messages

    if not messages:
        return ["Chưa có lịch sử chat nào trong Database của session này!"]

    readable_history = []

    for msg in history_obj.messages:
        readable_history.append({
            "Role": msg.type.upper(), #msg type could be ai or human role
            "Content": msg.content 
        })

    return readable_history