"""Eval retrieval-only của chain single-pass, đo context recall/precision (RAGAS) và hop-recall cho tập multi-hop.

Hop-recall là metric deterministic (không đụng judge): đo xem chain có kéo được
cả Điều thứ hai (second hop) từ cột `link` vào trong tập parent đã retrieve hay
không, slice theo `group` thay vì trung bình toàn tập
"""

import argparse
import asyncio
import json
import os
import pickle
import random
import re
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


_DIEU_PATTERN = re.compile(r"Điều\s+(\d+)")


def parse_link(link: str) -> tuple[set[int], int]:
    """Tách link dạng 'A -> B' thành (gold_dieu, second_hop).

    A là Điều nguồn, B là Điều thứ hai (second hop - phần khó).
    Raise nếu format sai, không nuốt lỗi im lặng
    """
    parts = [p.strip() for p in link.split("->")]
    if len(parts) != 2:
        raise ValueError(f"link không đúng định dạng 'A -> B': {link!r}")
    try:
        source, second = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"link chứa số không hợp lệ: {link!r}") from exc
    return {source, second}, second


def compute_hop_recall(gold_dieu: set[int], second_hop: int, retrieved_dieu: set[int]) -> tuple[float, bool]:
    """Metric thuần, không cần API call.

    Trả về (full_recall, second_hop_hit):
    - full_recall: tỉ lệ gold Điều xuất hiện trong retrieved_dieu (0..1)
    - second_hop_hit: Điều thứ hai B có nằm trong retrieved_dieu (con số quyết định)
    """
    if not gold_dieu:
        raise ValueError("gold_dieu rỗng")
    full_recall = len(gold_dieu & retrieved_dieu) / len(gold_dieu)
    return full_recall, second_hop in retrieved_dieu


def _extract_dieu_number(doc: Document) -> int | None:
    """Lấy số Điều của parent, ưu tiên metadata 'Điều' rồi fallback về title.

    Cả hai nguồn đều do splitter sinh ra ở format cố định 'Điều <số>. ...' -
    đã probe doc_store_pdr: parent metadata giữ key 'Điều' (giá trị là chuỗi
    tên Điều), chỉ doc preamble có None nên cần fallback title
    """
    candidates = [doc.metadata.get("Điều"), doc.metadata.get("title", "")]
    for source in candidates:
        if source:
            match = _DIEU_PATTERN.search(source)
            if match:
                return int(match.group(1))
    return None


class EvalInputRow(BaseModel):
    id: Any
    user_input: str
    response: str
    retrieved_contexts: list[str] | str = Field(default_factory=list)
    # Các field dưới chỉ tồn tại trong corpus_cross_references.json, default None để corpus.json vẫn parse được
    link: str | None = None
    group: str | None = None
    type: str | None = None


class EvalScores(BaseModel):
    context_recall: float
    context_precision: float


class HopScores(BaseModel):
    gold_dieu: set[int]
    retrieved_dieu: set[int]
    second_hop: int
    full_recall: float
    second_hop_hit: bool


class ExperimentResultRow(BaseModel):
    id: Any
    query: str
    predicted_response: str = ""
    reference: str
    retrieved_contexts: list[str]
    scores: EvalScores
    group: str | None = None
    hop_scores: HopScores | None = None


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
) -> tuple[list[str], set[int]]:
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

    #Expose số Điều ra ngoài trước khi stringify - hop-recall cần set này
    retrieved_dieu: set[int] = set()
    for doc in parent_docs:
        number = _extract_dieu_number(doc)
        if number is not None:
            retrieved_dieu.add(number)

    #Ghép title vào context để mirror production
    contexts = [
        f"{doc.metadata.get('title', '')}\n{doc.page_content}".strip()
        for doc in parent_docs
    ]
    return contexts, retrieved_dieu


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
            # Reasoning token của qwen3 đếm vào max_tokens: phải đủ chứa cả think lẫn verdict
            max_tokens=10000,
        ),
        cache=cacher,
    )

    context_precision_metric = ContextPrecision(llm=judge_llm)
    context_recall_metric = ContextRecall(llm=judge_llm)

    @experiment(ExperimentResultRow)
    async def run_retrieval_eval(row: EvalInputRow) -> ExperimentResultRow:
        retrieved_contexts, retrieved_dieu = retrieve_parent_contexts(
            query=row.user_input,
            rewrite_chain=rewrite_chain,
            ensemble_retriever=ensemble_retriever,
            reranker=reranker,
            doc_store=doc_store,
        )

        #Hop-recall chạy song song với RAGAS, chỉ tính khi row có link (multi-hop)
        hop_scores: HopScores | None = None
        if row.link:
            gold_dieu, second_hop = parse_link(row.link)
            full_recall, second_hop_hit = compute_hop_recall(gold_dieu, second_hop, retrieved_dieu)
            hop_scores = HopScores(
                gold_dieu=gold_dieu,
                retrieved_dieu=retrieved_dieu,
                second_hop=second_hop,
                full_recall=full_recall,
                second_hop_hit=second_hop_hit,
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
        if hop_scores is not None:
            print(
                f"id={row.id} group={row.group} gold={sorted(hop_scores.gold_dieu)} "
                f"retrieved={sorted(hop_scores.retrieved_dieu)} "
                f"full_recall={hop_scores.full_recall:.2f} "
                f"second_hop_hit={hop_scores.second_hop_hit}"
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
            group=row.group,
            hop_scores=hop_scores,
        )

    all_results: list[dict[str, Any]] = []
    for i, row in enumerate(dataset):
        exp_result = await run_retrieval_eval(row)
        parsed_rows = _normalize_experiment_results(exp_result)
        #mode="json" để set trong HopScores serialize được ra JSON
        all_results.extend([parsed.model_dump(mode="json") for parsed in parsed_rows])

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

    #Aggregate hop-recall slice theo group, không trung bình toàn tập
    hop_by_group: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in all_results:
        if isinstance(item, dict) and item.get("hop_scores") is not None:
            grouped.setdefault(item.get("group") or "unknown", []).append(item["hop_scores"])
    for group_name, hops in grouped.items():
        hop_by_group[group_name] = {
            "second_hop_hit_rate": round(sum(h["second_hop_hit"] for h in hops) / len(hops), 4),
            "mean_full_recall": round(sum(h["full_recall"] for h in hops) / len(hops), 4),
            "n": len(hops),
        }

    output = {
        "created_at": datetime.now(UTC).isoformat(),
        "aggregate_scores": aggregate_scores,
        "hop_recall_by_group": hop_by_group or None,
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
