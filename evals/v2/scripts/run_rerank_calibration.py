"""
Reranker threshold calibration script — Phase 1 (measurement only).

Purpose:
    Measure the real rerank_score distribution of gold docs vs. noise docs
    across ~25 single-hop + table queries, WITHOUT applying any threshold filter.
    Output is a scored CSV + histogram figure to inform floor/margin values.

Key differences from production (qa_chain.py):
    - NO query rewrite / llama-70b expansion (dataset is unambiguous single-hop/table).
      Hybrid retrieve runs directly on user_input. Scores reflect the original question.
      NOTE: label these numbers "no-rewrite" — do not mistake them for production numbers.
    - NO floor 0.5 filter — all candidates are kept with their scores.
    - NO parent fetch for generation — pipeline stops after child→parent ID mapping.

Gold matching:
    retrieved_contexts first line is "Điều N. ...", so gold Điều numbers are parsed
    via regex from that string. Candidates are matched via metadata["title"] which
    splitter.py:70 sets to "{Chương} - {Điều N. ...}" — regex Điều\\s+(\\d+) works
    on both sides. doc_id is the parent hash key, NOT the Điều number.

Usage:
    python -m evals.v2.scripts.run_rerank_calibration
    python -m evals.v2.scripts.run_rerank_calibration --dataset evals/datasets/corpus_reranker.json
    python -m evals.v2.scripts.run_rerank_calibration --no-figure
"""

import argparse
import csv
import json
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath("."))

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.storage import EncoderBackedStore, LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.rag.embedding_utils import get_embedding_model
from src.rag.reranker_utils import load_reranker

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CHROMA_PATH = "chroma_db"
DOC_STORE_PATH = "doc_store_pdr"
DEFAULT_DATASET_PATH = "evals/datasets/corpus_reranker.json"
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
    type: str = "single"


class CandidateRecord(BaseModel):
    query_id: Any
    query_type: str
    candidate_rank: int        # 1-based, sorted by rerank_score desc
    rerank_score: float
    dieu: str                  # e.g. "10" — parsed from metadata["title"]
    title: str                 # raw metadata["title"]
    is_gold: bool


# ---------------------------------------------------------------------------
# Model / retriever setup  (reused from run_evals_retrieval.py pattern)
# ---------------------------------------------------------------------------


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_retrievers(
    k: int, embedding_model
) -> tuple[EnsembleRetriever, EncoderBackedStore]:
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
# Gold parsing — extract Điều numbers from retrieved_contexts string
# ---------------------------------------------------------------------------

_DIEU_RE = re.compile(r"Điều\s+(\d+)", re.UNICODE)


def _parse_gold_dieu_numbers(retrieved_contexts: str | list[str]) -> set[str]:
    """Return the set of Điều numbers (as strings) that are gold for this query."""
    if isinstance(retrieved_contexts, list):
        text = " ".join(retrieved_contexts)
    else:
        text = retrieved_contexts
    return set(_DIEU_RE.findall(text))


def _extract_dieu_from_title(title: str) -> str:
    """
    metadata["title"] format: "{Chương X. ...} - {Điều N. ...}"
    Returns the Điều number string, or "" if not found.
    """
    m = _DIEU_RE.search(title)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Core retrieval + scoring  (NO floor, NO generation)
# ---------------------------------------------------------------------------


def score_candidates(
    question: str,
    ensemble_retriever: EnsembleRetriever,
    reranker: Any,
    gold_dieu: set[str],
) -> list[CandidateRecord]:
    """
    Hybrid retrieve → reranker.predict (original question, no rewrite) →
    return all candidates with scores, ranks, and gold flags.
    Floor 0.5 is intentionally NOT applied.
    """
    # Retrieve via single user_input (no expansion by design for this dataset)
    raw_docs: list[Document] = ensemble_retriever.invoke(question)

    # Deduplicate by page_content (same as production)
    dedup: dict[str, Document] = {}
    for doc in raw_docs:
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
                query_id=None,          # filled by caller
                query_type="",          # filled by caller
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


def _save_csv(records: list[CandidateRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "query_type", "candidate_rank", "rerank_score", "dieu", "title", "is_gold"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r.model_dump())
    print(f"[calibrate] CSV saved → {path}")


def _save_figure(records: list[CandidateRecord], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[calibrate] matplotlib not available — skipping figure generation.")
        return

    gold = [r for r in records if r.is_gold]
    noise = [r for r in records if not r.is_gold]

    type_colors = {"single": "#2196F3", "multi_hop": "#9C27B0", "table": "#FF9800"}
    type_labels = {"single": "single", "multi_hop": "multi-hop", "table": "table"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Reranker score distribution — gold vs. noise\n"
        "(no-rewrite run; floor 0.5 not applied)",
        fontsize=13,
        fontweight="bold",
    )

    bins = np.linspace(0.0, 1.0, 26)

    # ---- left panel: noise -------------------------------------------------
    ax_noise = axes[0]
    noise_by_type: dict[str, list[float]] = {}
    for r in noise:
        noise_by_type.setdefault(r.query_type, []).append(r.rerank_score)

    bottom = np.zeros(len(bins) - 1)
    for qtype, scores in sorted(noise_by_type.items()):
        counts, _ = np.histogram(scores, bins=bins)
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

    # ---- right panel: gold -------------------------------------------------
    ax_gold = axes[1]
    gold_by_type: dict[str, list[float]] = {}
    for r in gold:
        gold_by_type.setdefault(r.query_type, []).append(r.rerank_score)

    # queries whose gold score < 0.5 — annotate as "previously failed"
    failed_query_ids: set[Any] = set()
    gold_score_by_qid: dict[Any, float] = {}
    for r in gold:
        # keep the highest gold score per query (there may be multiple gold docs)
        if r.query_id not in gold_score_by_qid or r.rerank_score > gold_score_by_qid[r.query_id]:
            gold_score_by_qid[r.query_id] = r.rerank_score
    for qid, best_score in gold_score_by_qid.items():
        if best_score < 0.5:
            failed_query_ids.add(qid)

    bottom_g = np.zeros(len(bins) - 1)
    for qtype, scores in sorted(gold_by_type.items()):
        counts, _ = np.histogram(scores, bins=bins)
        ax_gold.bar(
            bins[:-1], counts, width=bins[1] - bins[0],
            bottom=bottom_g, align="edge",
            color=type_colors.get(qtype, "#888"),
            alpha=0.85, label=type_labels.get(qtype, qtype),
        )
        bottom_g = bottom_g + counts

    ax_gold.axvline(0.5, color="red", linestyle="--", linewidth=1.4, label="old floor 0.5")

    # Scatter the failed gold points at y=0 to make them visible
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

    # ---- summary text box --------------------------------------------------
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
    print(f"[calibrate] Figure saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_calibration(dataset_path: str, save_figure: bool = True) -> None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Dataset must be a JSON array.")

    dataset = [CalibInputRow.model_validate(row) for row in raw]
    print(f"[calibrate] Loaded {len(dataset)} queries from {dataset_path}")

    print("[calibrate] Loading embedding model…")
    embedding_model = get_embedding_model()

    print("[calibrate] Loading reranker…")
    reranker = load_reranker()

    print("[calibrate] Building retrievers…")
    ensemble_retriever, _ = _build_retrievers(k=RETRIEVER_K, embedding_model=embedding_model)

    all_records: list[CandidateRecord] = []

    for row in dataset:
        gold_dieu = _parse_gold_dieu_numbers(row.retrieved_contexts)
        print(
            f"[calibrate] id={row.id:>3} type={row.type:<10} "
            f"gold_dieu={sorted(gold_dieu)} | {row.user_input[:60]}"
        )

        records = score_candidates(
            question=row.user_input,
            ensemble_retriever=ensemble_retriever,
            reranker=reranker,
            gold_dieu=gold_dieu,
        )

        # Fill in query-level fields
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

    # Persist results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"rerank_calibration_{ts}.csv"
    _save_csv(all_records, csv_path)

    if save_figure:
        fig_path = RESULTS_DIR / f"rerank_calibration_{ts}.png"
        _save_figure(all_records, fig_path)

    # Quick summary to console
    gold_records = [r for r in all_records if r.is_gold]
    if gold_records:
        import statistics
        gold_scores = [r.rerank_score for r in gold_records]
        failed = [s for s in gold_scores if s < 0.5]
        print("\n[calibrate] ── Gold score summary ──")
        print(f"  n_gold_candidates : {len(gold_scores)}")
        print(f"  min               : {min(gold_scores):.4f}")
        print(f"  p5                : {sorted(gold_scores)[max(0, int(len(gold_scores)*0.05)-1)]:.4f}")
        print(f"  median            : {statistics.median(gold_scores):.4f}")
        print(f"  max               : {max(gold_scores):.4f}")
        print(f"  gold < 0.5        : {len(failed)} / {len(gold_scores)}")
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reranker calibration — no floor, no generation, no query rewrite."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help="Path to corpus_reranker.json",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_calibration(dataset_path=args.dataset, save_figure=not args.no_figure)
