"""
Reranker score calibration với query rewrite — mirrors production pipeline.

Mục tiêu: đo phân phối rerank_score của gold docs vs noise docs trên casual/
abbreviation phrasing, SAU query rewrite (llama-70b), KHÔNG áp floor.
Output dùng để set RERANK_FLOOR / RERANK_MARGIN trong src/rag/config.py.

Khác với run_rerank_calibration.py (no-rewrite baseline):
  - Query rewrite ĐƯỢC BẬT: [original] + sub_queries → ensemble.map().invoke()
  - Dataset mặc định: evals/datasets/reranker_abbreviation.json (user cung cấp)
  - Có rate-limit sleep giữa các query (rewrite dùng llama-70b)

Pipeline per query:
  rewrite → [original] + subqueries → ensemble.map().invoke() → dedup →
  reranker.predict(original_question, child) → NO floor → gold mapping → log

Gold match: regex Điều\\s+(\\d+) trên metadata["title"]
"""

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath("."))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.storage import EncoderBackedStore, LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random

from src.rag.embedding_utils import get_embedding_model
from src.rag.reranker_utils import load_reranker

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CHROMA_PATH = "chroma_db"
DOC_STORE_PATH = "doc_store_pdr"
DEFAULT_DATASET_PATH = "evals/datasets/reranker_abbreviation.json"
# A/B mode: hai corpus ghép cặp theo id (cùng câu hỏi, cùng gold Điều, khác văn phong)
FORMAL_DATASET_PATH = "evals/datasets/corpus_reranker.json"
ABBRE_DATASET_PATH = "evals/datasets/corpus_reranker_abbre.json"
RESULTS_DIR = Path("evals/v2/results")

RETRIEVER_K = 15
MAX_RERANK_CANDIDATES = 20

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CalibInputRow(BaseModel):
    id: Any
    user_input: str
    response: str = ""
    retrieved_contexts: str | list[str]
    type: str = "single"  # single | multi_hop | table


class CandidateRecord(BaseModel):
    query_id: Any
    query_type: str
    candidate_rank: int      # 1-based, sorted by rerank_score desc
    rerank_score: float
    dieu: str                # e.g. "10" — parsed from metadata["title"]
    title: str               # raw metadata["title"]
    is_gold: bool


# ---------------------------------------------------------------------------
# Model / retriever setup
# ---------------------------------------------------------------------------


def _build_retrievers(k: int, embedding_model) -> tuple[EnsembleRetriever, EncoderBackedStore]:
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


# ---------------------------------------------------------------------------
# Query rewrite (mirrors production)
# ---------------------------------------------------------------------------


class _QueryExpansion(BaseModel):
    reasoning: str = Field(description="Phân tích ngắn gọn ý định của câu hỏi gốc")
    queries: list[str] = Field(description="Danh sách tối đa 3 câu hỏi đơn lẻ bằng tiếng Việt")


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _build_rewrite_chain(llm: ChatGroq):
    parser = PydanticOutputParser(pydantic_object=_QueryExpansion)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Query Transformation Engine for a Vietnamese university regulation QA system.
Your ONLY Task: Given a user question, rewrite it into standalone Vietnamese sub-queries.

Rules:
- Output ONLY valid JSON matching the required schema.
- DO NOT answer the question.
- NEVER ask for clarification.
- Keep the original question intact as the first query if no rewrite needed.
- Preserve ALL Vietnamese legal/academic terms unchanged.
- Generate maximum 3 sub-queries.

{format_instructions}"""),
        ("human", "{question}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random(min=30, max=60),
    retry=retry_if_exception(_is_rate_limited),
    reraise=True,
)
def _rewrite_query(question: str, rewrite_chain) -> list[str]:
    try:
        parsed: _QueryExpansion = rewrite_chain.invoke({"question": question})
    except Exception as exc:
        if _is_rate_limited(exc):
            raise RuntimeError(str(exc)) from exc
        raise
    queries = [q.strip() for q in parsed.queries if isinstance(q, str) and q.strip()]
    return queries if queries else [question]


def _sleep_offset(label: str, min_s: int = 1, max_s: int = 2) -> None:
    s = random.randint(min_s, max_s)
    print(f"  [{label}] sleeping {s}s...")
    time.sleep(s)


# ---------------------------------------------------------------------------
# Gold parsing
# ---------------------------------------------------------------------------

_DIEU_RE = re.compile(r"Điều\s+(\d+)", re.UNICODE)


def _parse_gold_dieu_numbers(retrieved_contexts: str | list[str]) -> set[str]:
    if isinstance(retrieved_contexts, list):
        text = " ".join(retrieved_contexts)
    else:
        text = retrieved_contexts
    return set(_DIEU_RE.findall(text))


def _extract_dieu_from_title(title: str) -> str:
    m = _DIEU_RE.search(title)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Core retrieval + scoring  (NO floor)
# ---------------------------------------------------------------------------


def score_candidates(
    question: str,
    ensemble_retriever: EnsembleRetriever,
    reranker: Any,
    rewrite_chain,
    gold_dieu: set[str],
) -> list[CandidateRecord]:
    """
    Rewrite → [original] + subqueries → ensemble.map().invoke() → dedup →
    reranker.predict(original, child) → NO floor → return all with scores + gold flags.
    """
    _sleep_offset("rewrite")
    sub_queries = _rewrite_query(question, rewrite_chain)
    queries_to_retrieve = [question] + sub_queries

    nested: list[list[Document]] = ensemble_retriever.map().invoke(queries_to_retrieve)

    dedup: dict[str, Document] = {}
    for sublist in nested:
        for doc in sublist:
            dedup.setdefault(doc.page_content, doc)
    merged = list(dedup.values())[:MAX_RERANK_CANDIDATES]

    if not merged:
        return []

    pairs = [(question, doc.page_content) for doc in merged]
    scores = reranker.predict(pairs)

    scored = sorted(
        zip(merged, [float(s) for s in scores]),
        key=lambda x: x[1],
        reverse=True,
    )

    records: list[CandidateRecord] = []
    for rank, (doc, score) in enumerate(scored, start=1):
        title = doc.metadata.get("title", "")
        dieu = _extract_dieu_from_title(title)
        records.append(
            CandidateRecord(
                query_id=None,       # filled by caller
                query_type="",       # filled by caller
                candidate_rank=rank,
                rerank_score=round(score, 6),
                dieu=dieu,
                title=title,
                is_gold=(dieu in gold_dieu) if dieu else False,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _dump_json_atomic(payload, path: Path) -> None:
    """Ghi JSON atomic: dump ra file .tmp rồi rename. Tránh file NUL/nửa vời khi
    script chết giữa chừng (đã gặp: CSV ghi từng dòng bị đứt -> toàn byte \\x00)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _save_json(records: list[CandidateRecord], path: Path) -> None:
    """Single-mode: toàn bộ candidate + metadata, dạng JSON."""
    payload = {
        "created_at": datetime.now().isoformat(),
        "n_candidates": len(records),
        "candidates": [r.model_dump() for r in records],
    }
    _dump_json_atomic(payload, path)
    print(f"[calib] JSON saved → {path}")


def _save_figure(records: list[CandidateRecord], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[calib] matplotlib not available — skipping figure.")
        return

    gold = [r for r in records if r.is_gold]
    noise = [r for r in records if not r.is_gold]

    type_colors = {"single": "#2196F3", "multi_hop": "#9C27B0", "table": "#FF9800"}
    type_labels = {"single": "single", "multi_hop": "multi-hop", "table": "table"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Reranker score distribution — gold vs. noise\n"
        "(with query rewrite; floor 0.5 not applied)",
        fontsize=13,
        fontweight="bold",
    )

    bins = np.linspace(0.0, 1.0, 26)

    # left panel: noise
    ax_noise = axes[0]
    noise_by_type: dict[str, list[float]] = {}
    for r in noise:
        noise_by_type.setdefault(r.query_type, []).append(r.rerank_score)

    bottom = np.zeros(len(bins) - 1)
    for qtype, sc in sorted(noise_by_type.items()):
        counts, _ = np.histogram(sc, bins=bins)
        ax_noise.bar(
            bins[:-1], counts, width=bins[1] - bins[0],
            bottom=bottom, align="edge",
            color=type_colors.get(qtype, "#888"),
            alpha=0.75, label=type_labels.get(qtype, qtype),
        )
        bottom = bottom + counts
    ax_noise.axvline(0.5, color="red", linestyle="--", linewidth=1.4, label="old floor 0.5")
    ax_noise.set_title("Non-gold candidates")
    ax_noise.set_xlabel("rerank_score")
    ax_noise.set_ylabel("count")
    ax_noise.legend(fontsize=9)

    # right panel: gold
    ax_gold = axes[1]
    gold_by_type: dict[str, list[float]] = {}
    for r in gold:
        gold_by_type.setdefault(r.query_type, []).append(r.rerank_score)

    gold_score_by_qid: dict[Any, float] = {}
    for r in gold:
        if r.query_id not in gold_score_by_qid or r.rerank_score > gold_score_by_qid[r.query_id]:
            gold_score_by_qid[r.query_id] = r.rerank_score
    failed_query_ids = {qid for qid, best in gold_score_by_qid.items() if best < 0.5}

    bottom_g = np.zeros(len(bins) - 1)
    for qtype, sc in sorted(gold_by_type.items()):
        counts, _ = np.histogram(sc, bins=bins)
        ax_gold.bar(
            bins[:-1], counts, width=bins[1] - bins[0],
            bottom=bottom_g, align="edge",
            color=type_colors.get(qtype, "#888"),
            alpha=0.85, label=type_labels.get(qtype, qtype),
        )
        bottom_g = bottom_g + counts

    ax_gold.axvline(0.5, color="red", linestyle="--", linewidth=1.4, label="old floor 0.5")

    failed_gold = [r for r in gold if r.query_id in failed_query_ids]
    if failed_gold:
        ax_gold.scatter(
            [r.rerank_score for r in failed_gold],
            [-0.15] * len(failed_gold),
            marker="v", color="red", s=60, zorder=5,
            label=f"gold < 0.5 (prev. failed, n={len(failed_gold)})",
            clip_on=False,
        )

    ax_gold.set_title("Gold candidates")
    ax_gold.set_xlabel("rerank_score")
    ax_gold.set_ylabel("count")
    ax_gold.legend(fontsize=9)

    if gold:
        scores_arr = np.array([r.rerank_score for r in gold])
        summary = (
            f"Gold n={len(gold)}  "
            f"min={scores_arr.min():.3f}  "
            f"p5={np.percentile(scores_arr, 5):.3f}  "
            f"median={np.median(scores_arr):.3f}  "
            f"max={scores_arr.max():.3f}\n"
            f"Queries with gold < 0.5 (prev. failed): {len(failed_query_ids)}"
        )
        fig.text(
            0.5, 0.01, summary,
            ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="#BDBDBD"),
        )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[calib] Figure saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_calibration(dataset_path: str, save_figure: bool = True) -> None:
    if "GROQ_API_KEY" not in os.environ:
        raise EnvironmentError("GROQ_API_KEY is required (used for query rewrite via llama-70b)")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Dataset must be a JSON array.")

    dataset = [CalibInputRow.model_validate(row) for row in raw]
    print(f"[calib] Loaded {len(dataset)} queries from {dataset_path}")

    print("[calib] Loading embedding model…")
    embedding_model = get_embedding_model()

    print("[calib] Loading reranker…")
    reranker = load_reranker()

    print("[calib] Building retrievers…")
    ensemble_retriever, _ = _build_retrievers(k=RETRIEVER_K, embedding_model=embedding_model)

    print("[calib] Building query rewrite chain (llama-3.3-70b-versatile)…")
    rewrite_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, max_retries=0)
    rewrite_chain = _build_rewrite_chain(rewrite_llm)

    all_records: list[CandidateRecord] = []

    for i, row in enumerate(dataset):
        gold_dieu = _parse_gold_dieu_numbers(row.retrieved_contexts)
        print(
            f"[calib] id={row.id:>3} type={row.type:<10} "
            f"gold_dieu={sorted(gold_dieu)} | {row.user_input[:60]}"
        )

        records = score_candidates(
            question=row.user_input,
            ensemble_retriever=ensemble_retriever,
            reranker=reranker,
            rewrite_chain=rewrite_chain,
            gold_dieu=gold_dieu,
        )

        for r in records:
            r.query_id = row.id
            r.query_type = row.type

        gold_in_batch = [r for r in records if r.is_gold]
        if not gold_in_batch:
            print(
                f"  ⚠  No gold candidate found for id={row.id} "
                f"(gold_dieu={sorted(gold_dieu)}) — check title matching."
            )
        else:
            best = max(gold_in_batch, key=lambda r: r.rerank_score)
            flag = " ← BELOW 0.5 (prev. failed)" if best.rerank_score < 0.5 else ""
            print(
                f"  ✓ gold top score={best.rerank_score:.4f} "
                f"rank={best.candidate_rank} dieu={best.dieu}{flag}"
            )

        all_records.extend(records)

        if i < len(dataset) - 1:
            _sleep_offset("between_samples")

    # Persist results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"rewrite_calib_{ts}.json"
    _save_json(all_records, json_path)

    if save_figure:
        fig_path = RESULTS_DIR / f"rewrite_calib_{ts}.png"
        _save_figure(all_records, fig_path)

    # Console summary
    gold_records = [r for r in all_records if r.is_gold]
    if gold_records:
        import statistics
        gold_scores = [r.rerank_score for r in gold_records]
        failed = [s for s in gold_scores if s < 0.5]
        print("\n[calib] ── Gold score summary (with rewrite) ──")
        print(f"  n_gold_candidates : {len(gold_scores)}")
        print(f"  min               : {min(gold_scores):.4f}")
        p5_idx = max(0, int(len(gold_scores) * 0.05) - 1)
        print(f"  p5                : {sorted(gold_scores)[p5_idx]:.4f}")
        print(f"  median            : {statistics.median(gold_scores):.4f}")
        print(f"  max               : {max(gold_scores):.4f}")
        print(f"  gold < 0.5        : {len(failed)} / {len(gold_scores)}")
        print()
        print("[calib] Suggested next step:")
        print(f"  Set RERANK_FLOOR below {min(gold_scores):.3f} (gold min).")
        print("  Set RERANK_MARGIN from gap between gold tail and noise peak.")


# ---------------------------------------------------------------------------
# A/B mode — so cặp văn phong CHUẨN vs VIẾT TẮT trên cùng bộ câu hỏi
# ---------------------------------------------------------------------------
#
# Phương pháp (vì sao cần mode này):
#   run_calibration ở trên đo phân phối score của MỘT tập. Nó không trả lời được
#   "viết tắt làm mất bao nhiêu điểm", vì so hai tập rời sẽ lẫn cả khác biệt độ khó
#   của chính câu hỏi. Ở đây corpus_reranker (chuẩn) và corpus_reranker_abbre (viết
#   tắt) ghép cặp theo `id` — cùng câu hỏi, cùng gold Điều, chỉ khác cách diễn đạt —
#   nên delta = gold_top_formal - gold_top_abbre đo ĐÚNG tác động của nhiễu văn phong,
#   đã khử độ khó câu hỏi.
#
#   Chạy TUẦN TỰ: hết tập chuẩn rồi mới tới tập viết tắt, dùng chung reranker /
#   retriever / rewrite_chain đã load để hai bên cùng điều kiện. Vẫn KHÔNG áp floor —
#   ghi lại toàn bộ candidate + gold_top + noise_top + FLOOR/MARGIN suy ra mỗi bên.
#
# Chỉ số per-query (ghép theo id):
#   gold_top_*   điểm cao nhất trong candidate gold mỗi bên
#   delta        formal - abbre (dương = viết tắt tụt điểm)
#   gold_rank_*  thứ hạng gold tốt nhất (phát hiện tụt hạng dù điểm còn cao -> MARGIN cắt)
#   noise_top_*  điểm noise cao nhất (khe gold/noise hẹp lại -> FLOOR khó đặt)
#   margin_*     gold_top - noise_top; ÂM nghĩa noise vượt gold, floor không cứu nổi


def _summarize_records(records: list[CandidateRecord]) -> dict:
    """Rút danh sách candidate của 1 câu thành các chỉ số cho bảng so sánh."""
    if not records:
        return {"n_candidates": 0, "gold_top": None, "gold_rank": None,
                "noise_top": None, "top1_is_gold": False}

    gold = [r for r in records if r.is_gold]
    noise = [r for r in records if not r.is_gold]
    best_gold = max(gold, key=lambda r: r.rerank_score) if gold else None
    return {
        "n_candidates": len(records),
        "gold_top": best_gold.rerank_score if best_gold else None,
        "gold_rank": best_gold.candidate_rank if best_gold else None,
        "noise_top": max((r.rerank_score for r in noise), default=None),
        "top1_is_gold": bool(gold) and best_gold.candidate_rank == 1,
    }


def _run_variant(dataset, variant, ensemble_retriever, reranker, rewrite_chain):
    """Chạy hết 1 tập. Trả (summary theo id, toàn bộ candidate dạng dict để ghi CSV)."""
    print(f"\n{'=' * 70}\n[ab] variant: {variant}  ({len(dataset)} câu)\n{'=' * 70}")
    summaries: dict[Any, dict] = {}
    all_rows: list[dict] = []

    for i, row in enumerate(dataset):
        gold_dieu = _parse_gold_dieu_numbers(row.retrieved_contexts)
        print(f"[{variant}] id={row.id:>3} type={row.type:<8} gold={sorted(gold_dieu)} | {row.user_input[:55]}")

        records = score_candidates(
            question=row.user_input,
            ensemble_retriever=ensemble_retriever,
            reranker=reranker,
            rewrite_chain=rewrite_chain,
            gold_dieu=gold_dieu,
        )
        for r in records:
            r.query_id = row.id
            r.query_type = row.type
            d = r.model_dump()
            d["variant"] = variant
            all_rows.append(d)

        s = _summarize_records(records)
        s["type"] = row.type
        s["user_input"] = row.user_input
        summaries[row.id] = s

        if s["gold_top"] is None:
            print(f"  ⚠  không tìm thấy gold (gold={sorted(gold_dieu)})")
        else:
            nt = f" noise_top={s['noise_top']:.4f}" if s["noise_top"] is not None else ""
            print(f"  gold_top={s['gold_top']:.4f} rank={s['gold_rank']}{nt} n={s['n_candidates']}")

        if i < len(dataset) - 1:
            _sleep_offset("between_samples")

    return summaries, all_rows


def _build_pairs(formal: dict[Any, dict], abbre: dict[Any, dict]) -> list[dict]:
    """Ghép theo id, chỉ giữ id có ở CẢ hai tập. Suy ra delta + margin mỗi bên."""
    pairs: list[dict] = []
    for qid in sorted(set(formal) & set(abbre)):
        f, a = formal[qid], abbre[qid]
        gf, ga = f["gold_top"], a["gold_top"]
        pairs.append({
            "query_id": qid,
            "type": f["type"],
            "gold_top_formal": gf,
            "gold_top_abbre": ga,
            "delta": (gf - ga) if (gf is not None and ga is not None) else None,
            "gold_rank_formal": f["gold_rank"],
            "gold_rank_abbre": a["gold_rank"],
            "rank_drop": (a["gold_rank"] - f["gold_rank"])
                         if (f["gold_rank"] is not None and a["gold_rank"] is not None) else None,
            "noise_top_formal": f["noise_top"],
            "noise_top_abbre": a["noise_top"],
            # margin = khe gold-noise; ÂM = noise vượt gold -> floor vô dụng, phải sửa tầng query
            "margin_formal": (gf - f["noise_top"])
                             if (gf is not None and f["noise_top"] is not None) else None,
            "margin_abbre": (ga - a["noise_top"])
                            if (ga is not None and a["noise_top"] is not None) else None,
            "top1_is_gold_formal": f["top1_is_gold"],
            "top1_is_gold_abbre": a["top1_is_gold"],
            "user_input_formal": f["user_input"],
            "user_input_abbre": a["user_input"],
        })
    return pairs


def _save_ab_json(pairs: list[dict], candidates: list[dict], path: Path) -> None:
    """AB-mode: gộp pairs (bảng delta) + candidates (từng doc, có variant) vào 1 JSON.
    Sweep floor/ratio offline đọc từ key "candidates"; đối chiếu gold-survival từ "pairs"."""
    payload = {
        "created_at": datetime.now().isoformat(),
        "n_pairs": len(pairs),
        "n_candidates": len(candidates),
        "pairs": pairs,
        "candidates": candidates,
    }
    _dump_json_atomic(payload, path)
    print(f"[ab] JSON saved → {path}  (sweep floor/ratio offline từ key 'candidates')")


def _print_ab_report(pairs: list[dict]) -> None:
    import statistics
    deltas = [p["delta"] for p in pairs if p["delta"] is not None]
    ga = [p["gold_top_abbre"] for p in pairs if p["gold_top_abbre"] is not None]

    print(f"\n{'=' * 70}\n[ab] BÁO CÁO SO SÁNH VĂN PHONG\n{'=' * 70}")
    print(f"  số cặp ghép được    : {len(pairs)}")
    print(f"  cặp có delta        : {len(deltas)}")

    if deltas:
        worse = [d for d in deltas if d > 0]
        print(f"\n  --- DELTA (chuẩn - viết tắt), dương = viết tắt tụt điểm ---")
        print(f"    mean={statistics.mean(deltas):+.4f}  median={statistics.median(deltas):+.4f}")
        print(f"    min={min(deltas):+.4f}  max={max(deltas):+.4f}")
        if len(deltas) > 1:
            print(f"    stdev={statistics.stdev(deltas):.4f}")
        print(f"    số câu bị tụt điểm: {len(worse)}/{len(deltas)}")

    by_type: dict[str, list[float]] = {}
    for p in pairs:
        if p["delta"] is not None:
            by_type.setdefault(p["type"], []).append(p["delta"])
    if len(by_type) > 1:
        print(f"\n  --- DELTA theo type ---")
        for t, ds in sorted(by_type.items()):
            print(f"    {t:<8} n={len(ds):<3} mean={statistics.mean(ds):+.4f} max={max(ds):+.4f}")

    # MARGIN diagnostics: rank_drop cho biết margin có cắt nhầm gold không
    drops = [p for p in pairs if p["rank_drop"] is not None and p["rank_drop"] > 0]
    if drops:
        print(f"\n  --- gold TỤT HẠNG ở bản viết tắt (margin có thể cắt) ---")
        for p in sorted(drops, key=lambda p: -p["rank_drop"])[:8]:
            print(f"    id={p['query_id']:<3} rank {p['gold_rank_formal']} -> {p['gold_rank_abbre']} "
                  f"(drop {p['rank_drop']}), gold {p['gold_top_abbre']:.4f}")

    broken = [p for p in pairs if p["margin_abbre"] is not None and p["margin_abbre"] < 0]
    if broken:
        print(f"\n  ⚠  {len(broken)} câu NOISE VƯỢT GOLD ở bản viết tắt (floor không cứu nổi):")
        for p in broken:
            print(f"    id={p['query_id']} gold={p['gold_top_abbre']:.4f} noise={p['noise_top_abbre']:.4f}")
            print(f"      {p['user_input_abbre'][:65]}")

    missing = [p for p in pairs if p["gold_top_abbre"] is None]
    if missing:
        print(f"\n  ⚠  {len(missing)} câu KHÔNG retrieve được gold ở bản viết tắt:")
        for p in missing:
            print(f"    id={p['query_id']} | {p['user_input_abbre'][:65]}")

    if ga:
        print(f"\n  → RERANK_FLOOR phải THẤP HƠN {min(ga):.4f} (gold tệ nhất, bản viết tắt).")

    ranked = sorted((p for p in pairs if p["delta"] is not None), key=lambda p: -p["delta"])
    if ranked:
        print(f"\n  --- 5 câu tụt điểm nặng nhất ---")
        for p in ranked[:5]:
            print(f"    id={p['query_id']:<3} delta={p['delta']:+.4f} "
                  f"({p['gold_top_formal']:.4f} -> {p['gold_top_abbre']:.4f})")
            print(f"      {p['user_input_abbre'][:65]}")
    print()


def _save_ab_figure(pairs: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[ab] matplotlib chưa cài — bỏ qua biểu đồ.")
        return

    usable = [p for p in pairs if p["delta"] is not None]
    if not usable:
        print("[ab] không có cặp nào có delta — bỏ qua biểu đồ.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Ảnh hưởng văn phong viết tắt lên reranker score (ghép cặp theo id)",
                 fontsize=13, fontweight="bold")

    ids = [str(p["query_id"]) for p in usable]
    f_scores = [p["gold_top_formal"] for p in usable]
    a_scores = [p["gold_top_abbre"] for p in usable]
    deltas = [p["delta"] for p in usable]

    ax = axes[0]
    for fs, asc in zip(f_scores, a_scores):
        ax.plot([0, 1], [fs, asc], color="#B0BEC5", linewidth=1, zorder=1)
    ax.scatter([0] * len(f_scores), f_scores, color="#2196F3", s=45, zorder=3, label="chuẩn")
    ax.scatter([1] * len(a_scores), a_scores, color="#FF7043", s=45, zorder=3, label="viết tắt")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["chuẩn", "viết tắt"])
    ax.set_ylabel("gold_top score"); ax.set_title("Điểm gold theo từng câu")
    ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    colors = ["#EF5350" if d > 0 else "#66BB6A" for d in deltas]
    ax.bar(range(len(deltas)), deltas, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, fontsize=7, rotation=90)
    ax.set_xlabel("query_id"); ax.set_ylabel("delta (chuẩn - viết tắt)")
    ax.set_title("Mức tụt điểm mỗi câu\n(đỏ = viết tắt tệ hơn)"); ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    a_noise = [p["noise_top_abbre"] for p in usable if p["noise_top_abbre"] is not None]
    bins = np.linspace(0, 1, 26)
    ax.hist(a_noise, bins=bins, alpha=0.6, color="#78909C", label="noise_top (viết tắt)")
    ax.hist(a_scores, bins=bins, alpha=0.75, color="#FF7043", label="gold_top (viết tắt)")
    if a_scores:
        ax.axvline(min(a_scores), color="#D32F2F", linestyle="--", linewidth=1.5,
                   label=f"gold min = {min(a_scores):.3f}")
    ax.set_xlabel("rerank_score"); ax.set_ylabel("số câu")
    ax.set_title("Khe gold / noise (viết tắt)\nfloor phải dưới vạch đỏ"); ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ab] figure     → {path}")


def run_ab_calibration(formal_path: str, abbre_path: str, save_figure: bool = True) -> None:
    """Chạy tuần tự corpus chuẩn -> viết tắt, ghép cặp theo id, xuất bảng delta + margin."""
    if "GROQ_API_KEY" not in os.environ:
        raise EnvironmentError("GROQ_API_KEY is required (query rewrite via llama-70b)")

    def _load(p: str) -> list[CalibInputRow]:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"{p}: dataset phải là JSON array")
        return [CalibInputRow.model_validate(r) for r in raw]

    formal_ds, abbre_ds = _load(formal_path), _load(abbre_path)
    ids_f, ids_a = {r.id for r in formal_ds}, {r.id for r in abbre_ds}
    if ids_f != ids_a:
        print(f"[ab] ⚠  id hai tập không khớp — chỉ ghép phần chung.")
        print(f"     chỉ ở chuẩn   : {sorted(ids_f - ids_a)}")
        print(f"     chỉ ở viết tắt: {sorted(ids_a - ids_f)}")

    print(f"[ab] chuẩn   : {formal_path} ({len(formal_ds)} câu)")
    print(f"[ab] viết tắt: {abbre_path} ({len(abbre_ds)} câu)")

    print("[ab] loading embedding model…")
    embedding_model = get_embedding_model()
    print("[ab] loading reranker…")
    reranker = load_reranker()
    print("[ab] building retrievers…")
    ensemble_retriever, _ = _build_retrievers(k=RETRIEVER_K, embedding_model=embedding_model)
    print("[ab] building rewrite chain (llama-3.3-70b-versatile)…")
    rewrite_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, max_retries=0)
    rewrite_chain = _build_rewrite_chain(rewrite_llm)

    formal_sum, formal_rows = _run_variant(formal_ds, "formal", ensemble_retriever, reranker, rewrite_chain)
    _sleep_offset("chuyển variant")
    abbre_sum, abbre_rows = _run_variant(abbre_ds, "abbre", ensemble_retriever, reranker, rewrite_chain)

    pairs = _build_pairs(formal_sum, abbre_sum)
    _print_ab_report(pairs)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_ab_json(pairs, formal_rows + abbre_rows, RESULTS_DIR / f"ab_phrasing_{ts}.json")
    if save_figure:
        _save_ab_figure(pairs, RESULTS_DIR / f"ab_phrasing_{ts}.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reranker calibration with query rewrite — mirrors production pipeline."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help=f"[single mode] Path to dataset JSON (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--ab",
        action="store_true",
        help="A/B mode: chạy tuần tự corpus CHUẨN rồi corpus VIẾT TẮT (ghép cặp theo id) "
             "để đo delta điểm gold do văn phong gây ra. Xem run_ab_calibration.",
    )
    parser.add_argument(
        "--formal",
        default=FORMAL_DATASET_PATH,
        help=f"[ab mode] Corpus văn phong chuẩn (default: {FORMAL_DATASET_PATH})",
    )
    parser.add_argument(
        "--abbre",
        default=ABBRE_DATASET_PATH,
        help=f"[ab mode] Corpus viết tắt / lỗi chính tả (default: {ABBRE_DATASET_PATH})",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.ab:
        run_ab_calibration(
            formal_path=args.formal,
            abbre_path=args.abbre,
            save_figure=not args.no_figure,
        )
    else:
        run_calibration(dataset_path=args.dataset, save_figure=not args.no_figure)
