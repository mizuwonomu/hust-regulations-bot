import argparse
import asyncio
import json
import os
import pickle
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath('.'))

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.storage import EncoderBackedStore, LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from ragas import experiment
from ragas.cache import DiskCacheBackend as DiskCachedBackend
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ContextRecall
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random

#import từ config để eval tự bám theo production khi ratio đổi
from src.rag.config import (
    JUDGE_MODEL,
    QUERY_REWRITE_MODEL,
    RERANK_MAX_CHILDREN,
    RERANK_RATIO,
)
from src.rag.embedding_utils import get_embedding_model
from src.rag.reranker_utils import load_reranker

load_dotenv()

CHROMA_PATH = "chroma_db"
DOC_STORE_PATH = "doc_store_pdr"
DEFAULT_DATASET_PATH = "evals/datasets/corpus.json"
RAW_CACHE_DIR = Path("evals/v2/.raw_cache")


class QueryExpansion(BaseModel):
    reasoning: str = Field(description="Phân tích ngắn gọn ý định của câu hỏi gốc")
    queries: list[str] = Field(description="Danh sách 3 câu hỏi đơn lẻ bằng tiếng Việt để tìm kiếm")


class RateLimitError(RuntimeError):
    pass


class EvalInputRow(BaseModel):
    id: Any
    user_input: str
    response: str
    retrieved_contexts: list[str] | str = Field(default_factory=list)


class EvalScores(BaseModel):
    context_recall: float
    context_precision: float


class ExperimentResultRow(BaseModel):
    id: Any
    query: str
    predicted_response: str = ""
    reference: str
    retrieved_contexts: list[str]
    scores: EvalScores


def _random_offset_sleep(label: str, min_seconds: int = 1, max_seconds: int = 3) -> None:
    seconds = random.randint(min_seconds, max_seconds)
    print(f"[{label}] Sleeping {seconds}s for API offset policy...")
    time.sleep(seconds)


def _is_rate_limited_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return ("429" in message) or ("rate limit" in message) or ("too many requests" in message)


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_embedding_model() -> HuggingFaceEmbeddings:
    return get_embedding_model()


def _build_retrievers(k: int, embedding_model: HuggingFaceEmbeddings) -> tuple[EnsembleRetriever, EncoderBackedStore]:
    vector_store = Chroma(
        collection_name="split_parents",
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
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
        weights=[0.5, 0.5],
    )

    fs = LocalFileStore(DOC_STORE_PATH)
    doc_store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )

    return ensemble_retriever, doc_store


def _build_rewrite_chain(llm: ChatGroq):
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
    retry=retry_if_exception(_is_rate_limited_error),
    reraise=True,
)
def _rewrite_into_subqueries(question: str, rewrite_chain) -> list[str]:
    """
    Parse rewrite output into QueryExpansion and return the `queries` field.

    This intentionally mirrors `src/rag/qa_chain.py` behavior:
    - rely on the Pydantic model (`QueryExpansion`) as contract
    - if parsed `queries` is empty, fallback to `[original_question]`
    """
    try:
        parsed: QueryExpansion = rewrite_chain.invoke({"question": question})
    except Exception as exc:
        if _is_rate_limited_error(exc):
            raise RateLimitError(str(exc)) from exc
        raise

    queries = [q.strip() for q in parsed.queries if isinstance(q, str) and q.strip()]

    if not queries:
        return [question]

    return queries


def _rerank_ratio_filter(question: str, docs: list[Document], reranker: Any) -> list[Document]:
    """Mirror production: src/rag/qa_chain.py::_apply_score_ratio.

    top-1 giữ vô điều kiện, rank>=2 giữ nếu score >= top_score * RERANK_RATIO.
    KHÔNG có sàn tuyệt đối và KHÔNG có fallback top-k: production trả rỗng thì
    eval cũng phải trả rỗng
    """
    if not docs:
        return []

    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, [float(s) for s in scores]))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_score = scored_docs[0][1]
    cutoff = top_score * RERANK_RATIO

    kept = [scored_docs[0][0]]
    kept.extend(doc for doc, s in scored_docs[1:] if s >= cutoff)

    return kept[:RERANK_MAX_CHILDREN]


def retrieve_parent_contexts(
    query: str,
    rewrite_chain,
    ensemble_retriever: EnsembleRetriever,
    reranker: Any,
    doc_store: EncoderBackedStore,
) -> list[str]:
    # Required offset before each retrieval-only invoke (query rewrite uses qwen3)
    _random_offset_sleep(label="retrieval_invoke")

    sub_queries = _rewrite_into_subqueries(query, rewrite_chain)

    # Step 2: parallel retrieval for each sub-query (max 20 chunks each retriever)
    nested_docs: list[list[Document]] = ensemble_retriever.map().invoke(sub_queries)

    # Step 3: merge + deduplicate by content
    dedup_map: dict[str, Document] = {}
    for sublist in nested_docs:
        for doc in sublist:
            dedup_map.setdefault(doc.page_content, doc)

    merged_docs = list(dedup_map.values())

    # Step 4: rerank + chọn theo tỉ lệ tương đối với top-1 (khớp production)
    selected_children = _rerank_ratio_filter(query, merged_docs, reranker)

    # Step 5: fetch parent docs by parent IDs
    parent_ids: list[str] = []
    seen_ids = set()
    for doc in selected_children:
        p_id = doc.metadata.get("doc_id")
        if p_id and p_id not in seen_ids:
            seen_ids.add(p_id)
            parent_ids.append(p_id)

    parent_docs = [p for p in doc_store.mget(parent_ids) if p is not None]
    max_parents = 4
    if len(parent_docs) > max_parents:
        parent_docs = parent_docs[:max_parents]

    #Ghép title vào context để mirror production
    return [
        f"{doc.metadata.get('title', '')}\n{doc.page_content}".strip()
        for doc in parent_docs
    ]


def _normalize_experiment_results(exp_results: Any) -> list[ExperimentResultRow]:
    if isinstance(exp_results, list):
        normalized = exp_results
    elif hasattr(exp_results, "results"):
        normalized = []
        for item in exp_results.results:
            if hasattr(item, "output"):
                normalized.append(item.output)
            else:
                normalized.append(item)
    else:
        normalized = [exp_results]

    rows: list[ExperimentResultRow] = []
    for item in normalized:
        if isinstance(item, ExperimentResultRow):
            rows.append(item)
        elif isinstance(item, dict):
            rows.append(ExperimentResultRow.model_validate(item))
        else:
            rows.append(ExperimentResultRow.model_validate(item.model_dump()))

    return rows


async def run_eval(dataset_path: str, output_path: str, ratio: float | None = None) -> None:
    if "GROQ_API_KEY" not in os.environ:
        raise OSError("GROQ_API_KEY is required in environment or .env")

    #Cho phép quét ratio
    global RERANK_RATIO
    if ratio is not None:
        RERANK_RATIO = ratio
    print(f"[config] RERANK_RATIO = {RERANK_RATIO}")

    with open(dataset_path, "r", encoding="utf-8") as f:  # noqa: ASYNC230
        dataset_raw = json.load(f)

    if not isinstance(dataset_raw, list):
        raise ValueError("Dataset must be a JSON array of samples")  # noqa: TRY004
    dataset = [EvalInputRow.model_validate(row) for row in dataset_raw]

    embedding_model = _build_embedding_model()
    reranker = load_reranker()
    ensemble_retriever, doc_store = _build_retrievers(k=15, embedding_model=embedding_model)

    rewrite_llm = ChatGroq(
        model=QUERY_REWRITE_MODEL,
        temperature=0.2,
        max_retries=0,
        reasoning_effort="none",
    )
    rewrite_chain = _build_rewrite_chain(rewrite_llm)

    # Judge đi qua LangchainLLMWrapper + ChatGroq thay vì llm_factory
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cacher = DiskCachedBackend(cache_dir=str(RAW_CACHE_DIR))
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model=JUDGE_MODEL,
            temperature=0,
            max_retries=0,
            reasoning_format="parsed",
        ),
        cache=cacher,
    )

    context_precision_metric = ContextPrecision(llm=judge_llm)
    context_recall_metric = ContextRecall(llm=judge_llm)

    @experiment(ExperimentResultRow)
    async def run_retrieval_eval(row: EvalInputRow) -> ExperimentResultRow:
        retrieved_contexts = retrieve_parent_contexts(
            query=row.user_input,
            rewrite_chain=rewrite_chain,
            ensemble_retriever=ensemble_retriever,
            reranker=reranker,
            doc_store=doc_store,
        )

        # Metric legacy nhận SingleTurnSample thay vì kwargs như collections
        sample = SingleTurnSample(
            user_input=row.user_input,
            retrieved_contexts=retrieved_contexts,
            reference=row.response,
        )
        precision = float(await context_precision_metric.single_turn_ascore(sample))
        recall = float(await context_recall_metric.single_turn_ascore(sample))

        print(
            f"id={row.id} precision={precision:.4f} "
            f"recall={recall:.4f} contexts={len(retrieved_contexts)}"
        )

        return ExperimentResultRow(
            id=row.id,
            query=row.user_input,
            predicted_response="",
            reference=row.response,
            retrieved_contexts=retrieved_contexts,
            scores=EvalScores(
                context_recall=recall,
                context_precision=precision,
            ),
        )

    all_results: list[dict[str, Any]] = []
    for i, row in enumerate(dataset):
        exp_result = await run_retrieval_eval(row)
        parsed_rows = _normalize_experiment_results(exp_result)
        all_results.extend([parsed.model_dump() for parsed in parsed_rows])

        # Required delay between each sample evaluation
        if i < len(dataset) - 1:
            _random_offset_sleep(label="between_samples")

    metric_names = ("context_recall", "context_precision")
    aggregate_scores: dict[str, float | None] = {}
    for metric in metric_names:
        values = [
            float(item["scores"][metric])
            for item in all_results
            if isinstance(item, dict)
            and "scores" in item
            and isinstance(item["scores"], dict)
            and metric in item["scores"]
        ]
        aggregate_scores[metric] = round(sum(values) / len(values), 4) if values else None

    output = {
        "created_at": datetime.now(UTC).isoformat(),
        "aggregate_scores": aggregate_scores,
        "dataset_path": dataset_path,
        #Kết quả phải tự mô tả: hai run chỉ khác nhau ở ratio thì filename không đủ
        "config": {
            "rerank_ratio": RERANK_RATIO,
            "rerank_max_children": RERANK_MAX_CHILDREN,
            "retriever_k": 15,
            "max_parents": 4,
        },
        "num_samples": len(all_results),
        "results": all_results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved evaluation results to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval-only RAG evaluation with Ragas")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to corpus dataset JSON (default: evals/datasets/corpus.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"evals/v2/results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",  # noqa: DTZ005
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Override RERANK_RATIO cho lần chạy này (mặc định: lấy từ src/rag/config.py)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_eval(dataset_path=args.dataset, output_path=args.output, ratio=args.ratio))
