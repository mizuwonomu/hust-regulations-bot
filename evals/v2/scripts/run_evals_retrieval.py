"""Eval retrieval-only (single-pass hoặc citation agent), đo context recall/precision (RAGAS) và hop-recall cho tập multi-hop.

Hop-recall là nhóm metric deterministic (không đụng judge) trên cột `link` "A -> B",
tính trên tập Điều nên đảo thứ tự contexts không đổi điểm:
- gold_article_recall = |R ∩ G| / |G| - có lấy đủ Điều cần không
- gold_article_precision = |R ∩ G| / |R| - bao nhiêu Điều lấy về thuộc gold
  (tên gold_article_precision thay vì correctness để không nhầm với độ đúng câu trả lời)
- gold_article_f1 = 2PR / (P + R) - cân bằng thiếu và thừa
- all_gold_hit = G ⊆ R - có đủ toàn bộ Điều gold (tiêu chí retrieval chính)
- source_hit / target_hit: chẩn đoán đang thiếu Điều nguồn hay Điều đích
- missing_gold_recovery: trong những Điều gold baseline bỏ sót, agent lấy lại được bao nhiêu
Slice theo `group`, toàn tập báo tỷ lệ câu all_gold_hit=True.
Chế độ --agent kèm agent_trace: added_article_count so với seed thực tế,
new_gold_count, gate_calls, follow_count, termination_reason, context count
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath('.'))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from ragas import experiment
from ragas.cache import DiskCacheBackend as DiskCachedBackend
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ContextRecall

#Retrieval dùng chung với agent-exp capture; chỉ scoring/RAGAS ở lại module này
from evals.common.single_pass_retrieval import (
    QueryExpansion,
    RateLimitError,
    SinglePassSettings,
    build_retrievers,
    build_rewrite_chain,
    build_single_pass_runtime,
    random_offset_sleep,
    retrieve_parent_contexts,
    rewrite_into_subqueries,
)

#import từ config để eval tự bám theo production khi ratio đổi
from src.rag.agent.gate import decide_next
from src.rag.agent.llm_client import create_llm_client
from src.rag.agent.loop import agent_retrieve
from src.rag.agent.tools import (
    build_article_map,
    extract_citation_mentions,
    get_article,
)
from src.rag.config import JUDGE_MODEL

#Compat re-export cho run_ratio_sweep.py đang deferred: giữ đúng surface của HEAD
from src.rag.config import (  # noqa: F401
    QUERY_REWRITE_MODEL,
    RERANK_MAX_CHILDREN,
    RERANK_RATIO,
)

load_dotenv()


def __getattr__(name: str):
    """Re-export lazy cho sweep cũ, không kéo model loader vào lúc import module."""
    if name == "load_reranker":
        from src.rag.reranker_utils import load_reranker

        return load_reranker
    if name == "get_embedding_model":
        from src.rag.embedding_utils import get_embedding_model

        return get_embedding_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


CHROMA_PATH = "chroma_db"
DOC_STORE_PATH = "doc_store_pdr"
DEFAULT_DATASET_PATH = "evals/datasets/corpus.json"
RAW_CACHE_DIR = Path("evals/v2/.raw_cache")
AGENT_MAX_FOLLOW_ARTICLES = 3


def parse_link(link: str) -> tuple[int, int]:
    """Tách link dạng 'A -> B' thành (source_dieu, target_dieu).

    A là Điều nguồn, B là Điều đích được A viện dẫn - gold set là {A, B}.
    Raise nếu format sai, không nuốt lỗi im lặng
    """
    parts = [p.strip() for p in link.split("->")]
    if len(parts) != 2:
        raise ValueError(f"link không đúng định dạng 'A -> B': {link!r}")
    try:
        source, target = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"link chứa số không hợp lệ: {link!r}") from exc
    return source, target


def compute_hop_recall(
    gold_dieu: set[int],
    source_dieu: int,
    target_dieu: int,
    retrieved_dieu: set[int],
) -> tuple[float, float, float, bool, bool, bool]:
    """Metric thuần, không cần API call, tính trên tập Điều nên set-based.

    Trả về (gold_article_recall, gold_article_precision, gold_article_f1,
    all_gold_hit, source_hit, target_hit):
    - gold_article_recall = |R ∩ G| / |G| - có lấy đủ Điều cần không
    - gold_article_precision = |R ∩ G| / |R| - bao nhiêu Điều lấy về thuộc gold
    - gold_article_f1 = 2PR / (P + R) - cân bằng thiếu và thừa
    - all_gold_hit = G ⊆ R - có đủ toàn bộ Điều gold - tiêu chí retrieval chính
    - source_hit / target_hit: chẩn đoán đang thiếu Điều nguồn hay Điều đích
    R rỗng thì precision và f1 bằng 0.0 thay vì chia 0
    """
    if not gold_dieu:
        raise ValueError("gold_dieu rỗng")
    gold_article_recall = len(gold_dieu & retrieved_dieu) / len(gold_dieu)
    gold_article_precision = (
        len(gold_dieu & retrieved_dieu) / len(retrieved_dieu) if retrieved_dieu else 0.0
    )
    if gold_article_precision + gold_article_recall > 0:
        gold_article_f1 = (
            2
            * gold_article_precision
            * gold_article_recall
            / (gold_article_precision + gold_article_recall)
        )
    else:
        gold_article_f1 = 0.0
    return (
        gold_article_recall,
        gold_article_precision,
        gold_article_f1,
        gold_dieu <= retrieved_dieu,
        source_dieu in retrieved_dieu,
        target_dieu in retrieved_dieu,
    )


def compute_missing_gold_recovery(
    baseline_dieu: set[int],
    gold_dieu: set[int],
    retrieved_dieu: set[int],
) -> tuple[list[int], list[int]]:
    """Đo agent cứu lại bao nhiêu Điều gold mà baseline đã bỏ sót.

    Trả về (missing_gold, recovered) - missing_gold rỗng nghĩa là baseline
    đã đủ gold, recovery không áp dụng
    """
    missing_gold = sorted(gold_dieu - baseline_dieu)
    recovered = sorted(set(missing_gold) & retrieved_dieu)
    return missing_gold, recovered


def _load_baseline_retrieved_dieu(baseline_path: str) -> dict[Any, set[int]]:
    """Đọc retrieved_dieu theo id từ file kết quả baseline (run single-pass)"""
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    baseline: dict[Any, set[int]] = {}
    for item in data.get("results", []):
        hop = item.get("hop_scores")
        if hop is not None:
            baseline[item["id"]] = set(hop["retrieved_dieu"])
    return baseline


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


class MissingGoldRecovery(BaseModel):
    missing_gold: list[int]
    recovered: list[int]


class GateDecision(BaseModel):
    """Một lần gọi gate đã ghi nhận trong loop, dùng để soi hành vi gate"""

    call: int
    candidates: list[int]
    stop: bool
    follow_dieu: int | None


class AgentTrace(BaseModel):
    """Trajectory chạy thật của agent cho một câu, tách khỏi điểm số tập-based.

    added_article_count đếm so với seed thực tế của agent trong run này,
    không so với baseline file; new_gold_count là phần added thuộc gold
    """

    seed_dieu: list[int]
    added_dieu: list[int]
    added_article_count: int
    new_gold_count: int | None
    gate_calls: int
    follow_count: int
    termination_reason: str
    context_count: int
    gate_decisions: list[GateDecision]


class HopScores(BaseModel):
    gold_dieu: set[int]
    retrieved_dieu: set[int]
    source_dieu: int
    target_dieu: int
    gold_article_recall: float
    gold_article_precision: float
    gold_article_f1: float
    all_gold_hit: bool
    source_hit: bool
    target_hit: bool
    # None khi không có baseline hoặc baseline đã đủ gold
    missing_gold_recovery: MissingGoldRecovery | None = None


class ExperimentResultRow(BaseModel):
    id: Any
    query: str
    predicted_response: str = ""
    reference: str
    retrieved_contexts: list[str]
    scores: EvalScores
    group: str | None = None
    hop_scores: HopScores | None = None
    # Chỉ có trong chế độ --agent, ghi lại trajectory chạy thật
    agent_trace: AgentTrace | None = None


#Adapter mỏng giữ interface cũ cho các sweep script đang deferred
_random_offset_sleep = random_offset_sleep


def _build_embedding_model():
    """Adapter mỏng cho script sweep chưa chuyển sang shared runtime."""
    from src.rag.embedding_utils import get_embedding_model

    return get_embedding_model()


def _build_retrievers(k: int, embedding_model):
    """Adapter mỏng cho script sweep chưa chuyển sang shared runtime."""
    return build_retrievers(
        k=k,
        embedding_model=embedding_model,
        chroma_path=CHROMA_PATH,
        chroma_collection="split_parents",
        doc_store_path=DOC_STORE_PATH,
    )


def _build_rewrite_chain(llm):
    """Adapter mỏng cho script sweep chưa chuyển sang shared runtime."""
    return build_rewrite_chain(llm)


def _rewrite_into_subqueries(question: str, rewrite_chain) -> list[str]:
    """Adapter mỏng cho script sweep chưa chuyển sang shared runtime."""
    return rewrite_into_subqueries(question, rewrite_chain)


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


async def run_eval(
    dataset_path: str,
    output_path: str,
    ratio: float | None = None,
    use_agent: bool = False,
    baseline_path: str | None = None,
) -> None:
    if "GROQ_API_KEY" not in os.environ:
        raise OSError("GROQ_API_KEY is required in environment or .env")

    #Cho phép quét ratio cho riêng runtime này, không đụng global dùng chung
    settings = SinglePassSettings.effective(rerank_ratio=ratio)
    print(f"[config] RERANK_RATIO = {settings.rerank_ratio}")

    with open(dataset_path, "r", encoding="utf-8") as f:  # noqa: ASYNC230
        dataset_raw = json.load(f)

    if not isinstance(dataset_raw, list):
        raise ValueError("Dataset must be a JSON array of samples")  # noqa: TRY004
    dataset = [EvalInputRow.model_validate(row) for row in dataset_raw]

    runtime = build_single_pass_runtime(settings)
    rewrite_chain = runtime.rewrite_chain
    ensemble_retriever = runtime.ensemble_retriever
    reranker = runtime.reranker
    doc_store = runtime.doc_store
    vector_store = runtime.vector_store

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

    # Recovery so với baseline: chỉ tính khi được trỏ vào file kết quả single-pass
    baseline_dieu: dict[Any, set[int]] = {}
    if baseline_path:
        baseline_dieu = _load_baseline_retrieved_dieu(baseline_path)
        print(
            f"[config] baseline loaded: {len(baseline_dieu)} câu có hop_scores "
            f"từ {baseline_path}"
        )
    elif use_agent:
        print(
            "[config] cảnh báo: --agent mà thiếu --baseline nên "
            "missing_gold_recovery không tính"
        )

    agent_deps = None
    if use_agent:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        logging.getLogger("src.rag.agent").setLevel(logging.INFO)
        article_map = build_article_map(vector_store)
        agent_deps = {
            "retrieve_seed_fn": partial(
                retrieve_parent_contexts,
                rewrite_chain=rewrite_chain,
                ensemble_retriever=ensemble_retriever,
                reranker=reranker,
                doc_store=doc_store,
                settings=settings,
            ),
            "get_article_fn": partial(
                get_article,
                article_map=article_map,
                doc_store=doc_store,
            ),
            "extract_citations_fn": extract_citation_mentions,
            "decide_fn": partial(decide_next, client=create_llm_client()),
            "internal_dieu": set(article_map.keys()),
            "max_follow_articles": AGENT_MAX_FOLLOW_ARTICLES,
        }
    print(f"[config] agent mode = {use_agent}")
    if use_agent:
        print(
            "[config] agent cần llama-server chạy sẵn tại "
            "CITATION_AGENT_BASE_URL, "
            f"max follow articles = {AGENT_MAX_FOLLOW_ARTICLES}"
        )

    @experiment(ExperimentResultRow)
    async def run_retrieval_eval(row: EvalInputRow) -> ExperimentResultRow:
        agent_trace: AgentTrace | None = None
        if agent_deps is not None:
            telemetry: dict[str, Any] = {}
            retrieved_contexts, retrieved_dieu = agent_retrieve(
                row.user_input,
                telemetry=telemetry,
                **agent_deps,
            )
            # added tính trên seed thực tế của agent trong run này, không phải baseline file
            seed_dieu = set(telemetry["seed_dieu"])
            added_dieu = sorted(retrieved_dieu - seed_dieu)
            # Gold mới chỉ định nghĩa được khi row có link (multi-hop)
            gold_dieu_row = set(parse_link(row.link)) if row.link else None
            agent_trace = AgentTrace(
                seed_dieu=telemetry["seed_dieu"],
                added_dieu=added_dieu,
                added_article_count=len(added_dieu),
                new_gold_count=(
                    len(set(added_dieu) & gold_dieu_row)
                    if gold_dieu_row is not None
                    else None
                ),
                gate_calls=telemetry["gate_calls"],
                follow_count=telemetry["follow_count"],
                termination_reason=telemetry["termination_reason"],
                context_count=len(retrieved_contexts),
                gate_decisions=telemetry["gate_decisions"],
            )
        else:
            retrieved_contexts, retrieved_dieu = retrieve_parent_contexts(
                query=row.user_input,
                rewrite_chain=rewrite_chain,
                ensemble_retriever=ensemble_retriever,
                reranker=reranker,
                doc_store=doc_store,
                settings=settings,
            )

        #Hop-recall chạy song song với RAGAS, chỉ tính khi row có link (multi-hop)
        hop_scores: HopScores | None = None
        if row.link:
            source_dieu, target_dieu = parse_link(row.link)
            gold_dieu = {source_dieu, target_dieu}
            (
                gold_article_recall,
                gold_article_precision,
                gold_article_f1,
                all_gold_hit,
                source_hit,
                target_hit,
            ) = compute_hop_recall(gold_dieu, source_dieu, target_dieu, retrieved_dieu)
            # Recovery chỉ áp dụng khi baseline có kết quả cho id này và có bỏ sót gold
            recovery: MissingGoldRecovery | None = None
            if row.id in baseline_dieu:
                missing_gold, recovered = compute_missing_gold_recovery(
                    baseline_dieu[row.id], gold_dieu, retrieved_dieu
                )
                if missing_gold:
                    recovery = MissingGoldRecovery(
                        missing_gold=missing_gold, recovered=recovered
                    )
            hop_scores = HopScores(
                gold_dieu=gold_dieu,
                retrieved_dieu=retrieved_dieu,
                source_dieu=source_dieu,
                target_dieu=target_dieu,
                gold_article_recall=gold_article_recall,
                gold_article_precision=gold_article_precision,
                gold_article_f1=gold_article_f1,
                all_gold_hit=all_gold_hit,
                source_hit=source_hit,
                target_hit=target_hit,
                missing_gold_recovery=recovery,
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
                f"gold_article_recall={hop_scores.gold_article_recall:.2f} "
                f"gold_article_precision={hop_scores.gold_article_precision:.2f} "
                f"gold_article_f1={hop_scores.gold_article_f1:.2f} "
                f"all_gold_hit={hop_scores.all_gold_hit} "
                f"source_hit={hop_scores.source_hit} "
                f"target_hit={hop_scores.target_hit}"
            )
            if hop_scores.missing_gold_recovery is not None:
                rec = hop_scores.missing_gold_recovery
                print(
                    f"id={row.id} recovery: missing={rec.missing_gold} "
                    f"recovered={rec.recovered} "
                    f"full_recovery={set(rec.recovered) == set(rec.missing_gold)}"
                )

        # Soi riêng tác dụng của gate: mỗi lần gọi đã chọn gì trên candidates nào
        if agent_trace is not None:
            print(
                f"id={row.id} agent_trace: seed={agent_trace.seed_dieu} "
                f"added={agent_trace.added_dieu} "
                f"added_article_count={agent_trace.added_article_count} "
                f"new_gold_count={agent_trace.new_gold_count} "
                f"gate_calls={agent_trace.gate_calls} "
                f"follow_count={agent_trace.follow_count} "
                f"termination={agent_trace.termination_reason} "
                f"contexts={agent_trace.context_count}"
            )
            for gate_decision in agent_trace.gate_decisions:
                action = (
                    "stop"
                    if gate_decision.stop
                    else f"follow={gate_decision.follow_dieu}"
                )
                print(
                    f"id={row.id} gate soi: call={gate_decision.call} "
                    f"candidates={gate_decision.candidates} -> {action}"
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
            agent_trace=agent_trace,
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

    #Toàn tập: tỷ lệ câu all_gold_hit=True là tiêu chí chính của hop-recall
    hop_rows = [
        item["hop_scores"]
        for item in all_results
        if isinstance(item, dict) and item.get("hop_scores") is not None
    ]
    hop_overall: dict[str, Any] | None = None
    if hop_rows:
        hop_overall = {
            "all_gold_hit_rate": round(
                sum(h["all_gold_hit"] for h in hop_rows) / len(hop_rows), 4
            ),
            "mean_gold_article_recall": round(
                sum(h["gold_article_recall"] for h in hop_rows) / len(hop_rows), 4
            ),
            "mean_gold_article_precision": round(
                sum(h["gold_article_precision"] for h in hop_rows) / len(hop_rows), 4
            ),
            "mean_gold_article_f1": round(
                sum(h["gold_article_f1"] for h in hop_rows) / len(hop_rows), 4
            ),
            "n": len(hop_rows),
        }

    #Slice theo group: source/target_hit chẩn đoán đang thiếu phía nào
    hop_by_group: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in all_results:
        if isinstance(item, dict) and item.get("hop_scores") is not None:
            grouped.setdefault(item.get("group") or "unknown", []).append(item["hop_scores"])
    for group_name, hops in grouped.items():
        hop_by_group[group_name] = {
            "all_gold_hit_rate": round(sum(h["all_gold_hit"] for h in hops) / len(hops), 4),
            "mean_gold_article_recall": round(
                sum(h["gold_article_recall"] for h in hops) / len(hops), 4
            ),
            "mean_gold_article_precision": round(
                sum(h["gold_article_precision"] for h in hops) / len(hops), 4
            ),
            "mean_gold_article_f1": round(
                sum(h["gold_article_f1"] for h in hops) / len(hops), 4
            ),
            "source_hit_rate": round(sum(h["source_hit"] for h in hops) / len(hops), 4),
            "target_hit_rate": round(sum(h["target_hit"] for h in hops) / len(hops), 4),
            "n": len(hops),
        }

    #Recovery chỉ tính trên những câu baseline có bỏ sót gold
    recovery_rows = [
        h["missing_gold_recovery"]
        for h in hop_rows
        if h.get("missing_gold_recovery") is not None
    ]
    recovery_aggregate: dict[str, Any] | None = None
    if recovery_rows:
        n_full = sum(
            1
            for r in recovery_rows
            if set(r["recovered"]) == set(r["missing_gold"])
        )
        recovery_aggregate = {
            "n_applied": len(recovery_rows),
            "full_recovery_rate": round(n_full / len(recovery_rows), 4),
            "mean_recovery_rate": round(
                sum(
                    len(r["recovered"]) / len(r["missing_gold"])
                    for r in recovery_rows
                )
                / len(recovery_rows),
                4,
            ),
        }

    # Trajectory agent: tổng hợp gate, follow, tăng context và gold mới so với seed thật
    trace_rows = [
        item["agent_trace"]
        for item in all_results
        if isinstance(item, dict) and item.get("agent_trace") is not None
    ]
    agent_aggregate: dict[str, Any] | None = None
    if trace_rows:
        termination_counts: dict[str, int] = {}
        for trace in trace_rows:
            reason = trace["termination_reason"]
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
        agent_aggregate = {
            "total_gate_calls": sum(t["gate_calls"] for t in trace_rows),
            "gate_stop_decisions": sum(
                sum(1 for d in t["gate_decisions"] if d["stop"]) for t in trace_rows
            ),
            "total_follow_count": sum(t["follow_count"] for t in trace_rows),
            "termination_reasons": termination_counts,
            "total_added_article_count": sum(t["added_article_count"] for t in trace_rows),
            "total_new_gold_count": sum(
                t["new_gold_count"] for t in trace_rows if t["new_gold_count"] is not None
            ),
            "mean_context_count": round(
                sum(t["context_count"] for t in trace_rows) / len(trace_rows), 4
            ),
        }
        print(
            f"[gate analysis] gate_calls={agent_aggregate['total_gate_calls']} "
            f"stop_decisions={agent_aggregate['gate_stop_decisions']} "
            f"follows={agent_aggregate['total_follow_count']} "
            f"reasons={agent_aggregate['termination_reasons']} "
            f"added_articles={agent_aggregate['total_added_article_count']} "
            f"new_gold={agent_aggregate['total_new_gold_count']} "
            f"mean_contexts={agent_aggregate['mean_context_count']}"
        )

    output = {
        "created_at": datetime.now(UTC).isoformat(),
        "aggregate_scores": aggregate_scores,
        "hop_recall": {
            "overall": hop_overall,
            "by_group": hop_by_group or None,
            "missing_gold_recovery": recovery_aggregate,
        },
        "agent_trajectory": agent_aggregate,
        "baseline_path": baseline_path,
        "dataset_path": dataset_path,
        #Kết quả phải tự mô tả: hai run chỉ khác nhau ở ratio thì filename không đủ
        "config": {
            "rerank_ratio": settings.rerank_ratio,
            "rerank_max_children": settings.rerank_max_children,
            "retriever_k": settings.retriever_k,
            "max_parents": settings.max_parents,
            "agent": use_agent,
            "agent_max_follow_articles": (
                AGENT_MAX_FOLLOW_ARTICLES if use_agent else None
            ),
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
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Dùng citation agent qua llama.cpp thay cho single-pass retrieval",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path tới file kết quả run baseline (single-pass) để tính missing_gold_recovery",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_eval(
            dataset_path=args.dataset,
            output_path=args.output,
            ratio=args.ratio,
            use_agent=args.agent,
            baseline_path=args.baseline,
        )
    )
