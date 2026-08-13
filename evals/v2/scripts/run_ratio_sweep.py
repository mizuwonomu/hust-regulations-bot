"""
Sweep RERANK_RATIO trên nhiều corpus, ghi lại TOÀN BỘ scored list
"""

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

_spec = importlib.util.spec_from_file_location(
    "_eval_retrieval", _REPO / "evals/v2/scripts/run_evals_retrieval.py"
)
ev = importlib.util.module_from_spec(_spec)
sys.modules["_eval_retrieval"] = ev
_spec.loader.exec_module(ev)

CAP_CHILD = ev.RERANK_MAX_CHILDREN
CAP_PARENT = 4
_DIEU_RE = re.compile(r"Điều\s+(\d+)")

DATASETS = {
    "corpus": "evals/datasets/corpus.json",
    "crossref_multihop": "evals/datasets/corpus_cross_references.json",
    "reranker_formal": "evals/datasets/corpus_reranker.json",
    "reranker_abbre": "evals/datasets/corpus_reranker_abbre.json",
}
RATIOS = [round(0.60 - 0.05 * i, 2) for i in range(12)]  # 0.60 -> 0.05, bước đều


def gold_of(row) -> set[int]:
    ctx = row["retrieved_contexts"]
    ctx = ctx if isinstance(ctx, str) else " ".join(ctx)
    return {int(x) for x in _DIEU_RE.findall(ctx + " " + str(row.get("response", "")))}


def select_parents(scored, ratio):
    """scored: [(dieu, score)] đã sort desc. Trả (parents, n_kept)."""
    if not scored:
        return [], 0
    cutoff = scored[0][1] * ratio
    keep = [i for i, (_, s) in enumerate(scored) if s >= cutoff][:CAP_CHILD]
    parents, seen = [], set()
    for i in keep:
        d = scored[i][0]
        if d is not None and d not in seen:
            seen.add(d)
            parents.append(d)
    return parents[:CAP_PARENT], len(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(
        _REPO / f"evals/v2/results/ratio_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))
    args = ap.parse_args()

    emb = ev._build_embedding_model()
    reranker = ev.load_reranker()
    retriever, _ = ev._build_retrievers(k=15, embedding_model=emb)
    rewrite_chain = ev._build_rewrite_chain(
        ev.ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, max_retries=0)
    )

    raw = {}
    for name, path in DATASETS.items():
        rows = json.loads((_REPO / path).read_text(encoding="utf-8"))
        print(f"\n{'='*70}\n[{name}] {len(rows)} câu\n{'='*70}")
        recs = []
        for row in rows:
            gold = gold_of(row)
            ev._random_offset_sleep(label=f"{name}:{row['id']}")
            subs = ev._rewrite_into_subqueries(row["user_input"], rewrite_chain)
            dedup = {}
            for sub in retriever.map().invoke(subs):
                for d in sub:
                    dedup.setdefault(d.page_content, d)
            pool = list(dedup.values())
            if not pool:
                continue
            scores = [float(s) for s in reranker.predict([(row["user_input"], d.page_content) for d in pool])]
            scored = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
            # LƯU ĐỦ: rank -> Điều -> score -> gold, để sweep sau không cần API
            flat = [[d.metadata.get("dieu"), round(s, 6)] for d, s in scored]
            recs.append({"id": row["id"], "gold_dieu": sorted(gold), "scored": flat})
            print(f"  id={row['id']:>3} pool={len(pool)} top1={flat[0][1]:.4f} gold={sorted(gold)}")
        raw[name] = recs

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "scored list đầy đủ cho mọi câu; ratio sweep là phép tính offline trên dữ liệu này",
        "config": {"RERANK_MAX_CHILDREN": CAP_CHILD, "max_parents": CAP_PARENT, "retriever_k": 15},
        "ratios": RATIOS,
        "raw": raw,
    }, ensure_ascii=False), encoding="utf-8")

    # ---------- báo cáo ----------
    print("\n\n" + "=" * 88)
    print("GOLD RECALL / NHIỄU (parent không phải gold) THEO RATIO")
    print("=" * 88)
    header = f"{'ratio':<7}" + "".join(f"{n[:16]:>22}" for n in raw)
    print(header)
    print(f"{'':<7}" + "".join(f"{'recall  noise  kept':>22}" for _ in raw))
    print("-" * 88)
    for ratio in RATIOS:
        line = f"{ratio:<7.2f}"
        for name, recs in raw.items():
            hit = tot = 0
            noise = kept = 0
            for r in recs:
                parents, nk = select_parents(r["scored"], ratio)
                g = set(r["gold_dieu"])
                hit += len(g & set(parents)); tot += len(g)
                noise += len([p for p in parents if p not in g]); kept += nk
            line += f"{hit/tot:>9.3f}{noise/len(recs):>7.2f}{kept/len(recs):>6.1f}"
        print(line)

    print("\n" + "=" * 88)
    print("RATIO CHÍNH XÁC cần để cứu từng gold đang trượt ở 0.6 (knee thực nghiệm)")
    print("=" * 88)
    needed = []
    for name, recs in raw.items():
        for r in recs:
            parents, _ = select_parents(r["scored"], 0.6)
            top1 = r["scored"][0][1]
            for g in r["gold_dieu"]:
                if g in parents or top1 <= 0:
                    continue
                best = next((s for d, s in r["scored"] if d == g), None)
                if best is None:
                    continue
                needed.append((round(best / top1, 4), name, r["id"], g))
    needed.sort(reverse=True)
    for v, name, qid, g in needed:
        print(f"  ratio <= {v:<7.4f} | {name:<18} id={qid:<4} Điều{g}")
    if needed:
        vals = [v for v, *_ in needed]
        for th in (0.5, 0.4, 0.3, 0.25, 0.2, 0.1):
            print(f"  ratio {th}: cứu được {sum(1 for v in vals if v >= th)}/{len(vals)} gold đang trượt")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
