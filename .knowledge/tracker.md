# Tracker — HUST Regulations Bot

## Current focus  (2026-08-07)
Re-run e2e eval to confirm the applied rerank ratio 0.6 (expect id=13/15 improved; id=11/22/4/19 still red)  -> features/rerank_ratio/log.md

## Features

- rerank_ratio: DONE (2026-08-07, commit 9f48018) — shipped, e2e verification still outstanding  -> features/rerank_ratio/log.md

## Known limitations / debt left open

- **e2e not re-measured after applying ratio 0.6.** The code now runs `_apply_score_ratio`, but the expected gains (id=13/15) are predicted from the calibration sweep, not from an e2e run.
- **Negative reranker scores invert the ratio rule.** `cutoff = top_score * ratio` only orders correctly for `top_score > 0`; if top-1 scores negative the cutoff sits *above* it and only top-1 survives. Not observed on the current corpus (bge-reranker-v2-m3 sigmoid output), so left unguarded — revisit if the reranker model or its score range changes.
- **The calibration script is a frozen snapshot, not a repeatable mirror.** `run_rewrite_rerank_calibration.py` keeps `MAX_RERANK_CANDIDATES = 20` and a hardcoded `RETRIEVER_K`, so it reflects production *as of 2026-08-06* — before `fab3e07` removed the pre-rerank top-20 truncation. That is deliberate: freezing it is what keeps `ab_phrasing_20260806_163755.json` reproducible from the committed code. It does mean the script is NOT the "eval mirrors production" contract that applies to `evals/v2` retrieval eval. Before any future ratio re-sweep, re-align it with the current chain first, otherwise the new numbers describe a pipeline that no longer exists.
- **Out-of-scope detection has no owner.** Dropping the floor means the reranker no longer rejects off-topic queries ("giá vàng hôm nay"); "empty context" now fires only on a zero-candidate pool, which hybrid retrieval almost never produces. Needs the router (extend 'chat'/'RAG' to include out-of-scope) or the answer prompt to carry it. Currently unhandled.
- **Wrong-degree-level retrieval (id=13/15/17/21) is unsolved.** Ratio 0.6 mitigates it only when the gold twin is already rank-2 and near-tied (id=13/15); when gold is rank-6 or the column collapses (id=11/22) nothing in the rerank layer helps. The real fix — `bậc`+`chương` metadata at indexing + degree-selection before the chain + soft chapter-union — is designed only in discussion, requires re-ingestion, and the "union Chương I ∪ query chapter" heuristic is unverified (chapter-frequency in multi-hop gold was never measured).
- **Abbreviation-collapse (id=4/11) and table-comprehension (id=19) are out of scope here** and deferred to query-normalization and parser/splitter layers respectively.
- **Calibration set is 26 paired questions.** The worst-case gold tail that sets the ratio knee is effectively a single sample; 0.6 is a defensible knee, not a proven optimum. Re-sweep when the corpus grows, retrieval changes, or the reranker model changes.
- **Multi-hop and numeric-specific query types are underrepresented** in the corpus (only single + table); their score distributions are unmeasured, so the ratio's behaviour on them is an extrapolation.
