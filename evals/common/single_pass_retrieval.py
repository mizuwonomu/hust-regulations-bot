"""Single-pass retrieval dùng chung cho eval v2 và agent-exp seed capture.

Module giữ nguyên hành vi retrieval một lượt của eval: rewrite câu hỏi, hybrid
retrieval, rerank theo tỉ lệ top-1, rồi dựng lại parent context. Import nặng
(Chroma, provider LLM, model) chỉ chạy bên trong live runtime construction.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random

from src.rag.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    DOC_STORE_PATH,
    EMBEDDING_MODEL,
    QUERY_REWRITE_MODEL,
    QUERY_REWRITE_TEMPERATURE,
    RERANK_MAX_CHILDREN,
    RERANK_RATIO,
    RERANKER_MODEL,
    RETRIEVER_TOP_K,
)

if TYPE_CHECKING:
    from collections.abc import Callable

HYBRID_WEIGHTS: tuple[float, float] = (0.5, 0.5)
#Giữ tên cũ để call site cũ vẫn đọc được cùng semantics
DEFAULT_MAX_PARENTS = 4
_DIEU_PATTERN = re.compile(r"Điều\s+(\d+)")


def unsupported_model_override(settings: SinglePassSettings) -> str | None:
    """Trả lý do khi settings ghi tên model mà loader hiện tại không hề nhận."""
    if settings.embedding_model != EMBEDDING_MODEL:
        return (
            "embedding_model override is not supported by the current loader: "
            f"{settings.embedding_model!r} != {EMBEDDING_MODEL!r}"
        )
    if settings.reranker_model != RERANKER_MODEL:
        return (
            "reranker_model override is not supported by the current loader: "
            f"{settings.reranker_model!r} != {RERANKER_MODEL!r}"
        )
    return None


class QueryExpansion(BaseModel):
    reasoning: str = Field(description="Phân tích ngắn gọn ý định của câu hỏi gốc")
    queries: list[str] = Field(description="Danh sách 3 câu hỏi đơn lẻ bằng tiếng Việt để tìm kiếm")


class RateLimitError(RuntimeError):
    pass


class SinglePassSettings(BaseModel):
    """Cấu hình bất biến cho một single-pass retrieval runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retriever_k: int = Field(default=RETRIEVER_TOP_K, gt=0)
    hybrid_weights: tuple[float, float] = HYBRID_WEIGHTS
    rerank_ratio: float = Field(default=RERANK_RATIO, ge=0)
    rerank_max_children: int = Field(default=RERANK_MAX_CHILDREN, gt=0)
    max_parents: int = Field(default=DEFAULT_MAX_PARENTS, gt=0)
    rewrite_model: str = QUERY_REWRITE_MODEL
    rewrite_temperature: float = QUERY_REWRITE_TEMPERATURE
    rewrite_max_retries: int = 0
    rewrite_reasoning_effort: str = "none"
    embedding_model: str = EMBEDDING_MODEL
    reranker_model: str = RERANKER_MODEL
    chroma_path: str = CHROMA_PATH
    chroma_collection: str = CHROMA_COLLECTION
    doc_store_path: str = DOC_STORE_PATH

    @classmethod
    def effective(cls, *, rerank_ratio: float | None = None) -> SinglePassSettings:
        """Dựng settings từ default của config, chỉ override ratio khi được yêu cầu."""
        if rerank_ratio is None:
            return cls()
        return cls(rerank_ratio=rerank_ratio)

    @model_validator(mode="after")
    def _model_identity_is_honored(self) -> SinglePassSettings:
        reason = unsupported_model_override(self)
        if reason:
            raise ValueError(reason)
        return self


def random_offset_sleep(label: str, min_seconds: int = 1, max_seconds: int = 3) -> None:
    """Nghỉ ngẫu nhiên ngắn trước một lời gọi API để tránh dồn request."""
    seconds = random.randint(min_seconds, max_seconds)
    print(f"[{label}] Sleeping {seconds}s for API offset policy...")
    time.sleep(seconds)


def is_rate_limited_error(exc: Exception) -> bool:
    """Nhận diện lỗi rate limit của provider qua nội dung message."""
    message = str(exc).lower()
    return ("429" in message) or ("rate limit" in message) or ("too many requests" in message)


def extract_dieu_number(doc: Any) -> int | None:
    """Lấy số Điều của parent, ưu tiên metadata 'Điều' rồi fallback về title."""
    pattern = _DIEU_PATTERN
    candidates = [doc.metadata.get("Điều"), doc.metadata.get("title", "")]
    for source in candidates:
        if source:
            match = pattern.search(source)
            if match:
                return int(match.group(1))
    return None


def build_retrievers(
    *,
    k: int,
    embedding_model: Any,
    weights: tuple[float, float] = HYBRID_WEIGHTS,
    chroma_path: str = CHROMA_PATH,
    chroma_collection: str = CHROMA_COLLECTION,
    doc_store_path: str = DOC_STORE_PATH,
) -> tuple[Any, Any, Any]:
    """Dựng hybrid retriever, parent doc store và vector store từ store đã ingest."""
    import pickle

    from langchain_chroma import Chroma
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.storage import EncoderBackedStore, LocalFileStore
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    vector_store = Chroma(
        collection_name=chroma_collection,
        embedding_function=embedding_model,
        persist_directory=chroma_path,
    )

    child_vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    child_data = vector_store.get()
    all_child_docs = [
        Document(page_content=txt, metadata=md)
        for txt, md in zip(child_data["documents"], child_data["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(all_child_docs)
    bm25_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[child_vector_retriever, bm25_retriever],
        weights=list(weights),
    )

    fs = LocalFileStore(doc_store_path)
    doc_store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )

    return ensemble_retriever, doc_store, vector_store


def build_rewrite_chain(llm: Any):
    """Dựng prompt rewrite cùng output parser cho một LLM đã cấu hình."""
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    parser = PydanticOutputParser(pydantic_object=QueryExpansion)

    rephrase_system_prompt = """You are a Query Transformation Engine for a Vietnamese university regulation QA system.
    Your ONLY Task: Given a new user question, rewrite the question into standalone Vietnamese sub-queries.

    Rules:
    - Output ONLY valid JSON that follows the required schema.
    - DO NOT answer human's question.
    - NEVER ask for clarification.
    - If no rewrite needed, keep the original question text intact in the first query.
    - Preserve ALL Vietnamese legal/academic terms unchanged.
    - Generate maximum 3 sub-queries.

    {format_instructions}
    Examples:
    [No history] Query: "Quy định về học phí" -> Quy định về học phí
    [History: Quy định về học phí] Query: "Thế còn miễn giảm?" -> Quy định miễn giảm học phí tại HUST là gì?
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rephrase_system_prompt),
            ("human", "{question}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random(min=30, max=60),
    retry=retry_if_exception(is_rate_limited_error),
    reraise=True,
)
def rewrite_into_subqueries(question: str, rewrite_chain: Any) -> list[str]:
    """Parse rewrite output thành danh sách sub-query, fallback về câu hỏi gốc."""
    try:
        parsed: QueryExpansion = rewrite_chain.invoke({"question": question})
    except Exception as exc:
        if is_rate_limited_error(exc):
            raise RateLimitError(str(exc)) from exc
        raise

    queries = [q.strip() for q in parsed.queries if isinstance(q, str) and q.strip()]

    if not queries:
        return [question]

    return queries


def rerank_ratio_filter(
    question: str,
    docs: list[Any],
    reranker: Any,
    *,
    ratio: float,
    max_children: int,
) -> list[Any]:
    """Giữ top-1 vô điều kiện và các rank sau nếu score >= top_score * ratio."""
    if not docs:
        return []

    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, [float(s) for s in scores]))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_score = scored_docs[0][1]
    cutoff = top_score * ratio

    kept = [scored_docs[0][0]]
    kept.extend(doc for doc, s in scored_docs[1:] if s >= cutoff)

    return kept[:max_children]


def retrieve_parent_contexts(
    query: str,
    rewrite_chain: Any,
    ensemble_retriever: Any,
    reranker: Any,
    doc_store: Any,
    *,
    settings: SinglePassSettings,
    sleeper: Callable[[str], None] | None = None,
) -> tuple[list[str], set[int]]:
    """Chạy một lượt retrieval đầy đủ và trả về context đã render cùng tập Điều."""
    sleep = sleeper or random_offset_sleep
    # Required offset before each retrieval-only invoke (query rewrite uses qwen3)
    sleep("retrieval_invoke")

    sub_queries = rewrite_into_subqueries(query, rewrite_chain)

    # Step 2: retrieval cho từng sub-query rồi merge theo thứ tự xuất hiện
    nested_docs: list[list[Any]] = ensemble_retriever.map().invoke(sub_queries)

    # Step 3: merge + deduplicate by content
    dedup_map: dict[str, Any] = {}
    for sublist in nested_docs:
        for doc in sublist:
            dedup_map.setdefault(doc.page_content, doc)

    merged_docs = list(dedup_map.values())

    # Step 4: rerank + chọn theo tỉ lệ tương đối với top-1 (khớp production)
    selected_children = rerank_ratio_filter(
        query,
        merged_docs,
        reranker,
        ratio=settings.rerank_ratio,
        max_children=settings.rerank_max_children,
    )

    # Step 5: fetch parent docs by parent IDs
    parent_ids: list[str] = []
    seen_ids = set()
    for doc in selected_children:
        p_id = doc.metadata.get("doc_id")
        if p_id and p_id not in seen_ids:
            seen_ids.add(p_id)
            parent_ids.append(p_id)

    parent_docs = [p for p in doc_store.mget(parent_ids) if p is not None]
    if len(parent_docs) > settings.max_parents:
        parent_docs = parent_docs[: settings.max_parents]

    #Expose số Điều ra ngoài trước khi stringify - hop-recall cần set này
    retrieved_dieu: set[int] = set()
    for doc in parent_docs:
        number = extract_dieu_number(doc)
        if number is not None:
            retrieved_dieu.add(number)

    #Ghép title vào context để mirror production
    contexts = [
        f"{doc.metadata.get('title', '')}\n{doc.page_content}".strip()
        for doc in parent_docs
    ]
    return contexts, retrieved_dieu


@dataclass(frozen=True)
class SinglePassRuntime:
    """Giữ dependency đã dựng sẵn và cung cấp một seam retrieve duy nhất."""

    settings: SinglePassSettings
    rewrite_chain: Any
    ensemble_retriever: Any
    reranker: Any
    doc_store: Any
    vector_store: Any
    sleeper: Callable[[str], None] = random_offset_sleep
    metadata: dict[str, Any] = field(default_factory=dict)

    def retrieve(self, question: str) -> tuple[list[str], set[int]]:
        """Chạy single-pass retrieval cho một câu hỏi."""
        return retrieve_parent_contexts(
            question,
            self.rewrite_chain,
            self.ensemble_retriever,
            self.reranker,
            self.doc_store,
            settings=self.settings,
            sleeper=self.sleeper,
        )


def _default_rewrite_chain_builder(settings: SinglePassSettings):
    """Dựng rewrite chain từ provider thật; import provider chỉ chạy ở đây."""
    from langchain_groq import ChatGroq

    rewrite_llm = ChatGroq(
        model=settings.rewrite_model,
        temperature=settings.rewrite_temperature,
        max_retries=settings.rewrite_max_retries,
        reasoning_effort=settings.rewrite_reasoning_effort,
    )
    return build_rewrite_chain(rewrite_llm)


def build_single_pass_runtime(
    settings: SinglePassSettings,
    *,
    embedding_loader: Callable[[], Any] | None = None,
    reranker_loader: Callable[[], Any] | None = None,
    retriever_builder: Callable[..., tuple[Any, Any, Any]] | None = None,
    rewrite_chain_builder: Callable[[SinglePassSettings], Any] | None = None,
) -> SinglePassRuntime:
    """Dựng live runtime một lần cho toàn bộ capture hoặc eval run.

    Các loader/builder được inject để test offline không cần model thật; mặc định
    dùng đúng factory của repo.
    """
    #Chặn cả settings dựng qua model_construct để metadata không ghi tên model sai
    reason = unsupported_model_override(settings)
    if reason:
        raise ValueError(reason)

    if embedding_loader is None:
        from src.rag.embedding_utils import get_embedding_model

        embedding_loader = get_embedding_model
    if reranker_loader is None:
        from src.rag.reranker_utils import load_reranker

        reranker_loader = load_reranker
    if retriever_builder is None:
        retriever_builder = build_retrievers
    if rewrite_chain_builder is None:
        rewrite_chain_builder = _default_rewrite_chain_builder

    embedding_model = embedding_loader()
    reranker = reranker_loader()
    ensemble_retriever, doc_store, vector_store = retriever_builder(
        k=settings.retriever_k,
        embedding_model=embedding_model,
        weights=settings.hybrid_weights,
        chroma_path=settings.chroma_path,
        chroma_collection=settings.chroma_collection,
        doc_store_path=settings.doc_store_path,
    )
    rewrite_chain = rewrite_chain_builder(settings)

    #Metadata lấy model id từ config - nguồn thật mà loader dùng, không lấy lời khai settings
    metadata = settings.model_dump(mode="json")
    metadata["embedding_model"] = EMBEDDING_MODEL
    metadata["reranker_model"] = RERANKER_MODEL

    return SinglePassRuntime(
        settings=settings,
        rewrite_chain=rewrite_chain,
        ensemble_retriever=ensemble_retriever,
        reranker=reranker,
        doc_store=doc_store,
        vector_store=vector_store,
        metadata=metadata,
    )


__all__ = [
    "HYBRID_WEIGHTS",
    "QueryExpansion",
    "RateLimitError",
    "SinglePassRuntime",
    "SinglePassSettings",
    "build_retrievers",
    "build_rewrite_chain",
    "build_single_pass_runtime",
    "extract_dieu_number",
    "is_rate_limited_error",
    "random_offset_sleep",
    "rerank_ratio_filter",
    "retrieve_parent_contexts",
    "rewrite_into_subqueries",
    "unsupported_model_override",
]
