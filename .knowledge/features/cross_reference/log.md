# cross_reference — Log

## Narrative

The feature inherited a clear brief from the 2026-08-12 handoff: multi-hop questions whose gold articles sit at different degree levels retrieve the wrong level, and the regulation's own citation graph should be able to fix it. Two ingestion modules were already written — `reference_parser.py` extracts 27 citation edges from the Markdown (classifying each as `dependency` vs `applicability`, and `same_chapter` vs `cross_chapter`), and `degree_map.py` derives per-Điều degree scope by taking the chapter as a base layer and letting dependency edges extend a target with the borrowing article's level. `splitter.build_child_metadata` stamps the resulting `dieu`, `chuong` and four `ap_dung_*` booleans onto every child. The store had already been re-ingested with this metadata; `page_content` was deliberately untouched so the vectors were unchanged.

The session opened on the question of whether that work was safe to commit given nothing had been measured. Establishing that took an unexpected detour: the committed baseline `eval_retrieval_ratio_20260708.json` turned out to be unreproducible. Its contexts carry an "Điều-only" prefix, but the committed eval code returns bare `page_content` and the pending fix returns `Chương - Điều` — no version in git produces what the file contains, so it had been generated from uncommitted local edits. That artifact was retired as a baseline, and the rule that every measurement script must live in `evals/v2/scripts/` (never a scratch directory) was adopted for the rest of the work.

Design then converged on **anchor-filter** rather than pool-filter: instead of removing out-of-degree candidates, let the ratio's anchor be the first *in-degree* child, so an out-of-degree document loses the right to define the cutoff but is still kept if it clears it. This was chosen over a soft score boost because the project's founding constraint — reranker scores are not comparable across queries — forbids arithmetic on the score column; changing *which index* supplies the anchor does not touch the scores. A pool filter via a Chroma `where` clause was rejected because BM25 has no metadata filter and would leave the pool half-filtered. One design bug was caught during discussion: the existing code places the anchor at position 0 of the kept list, so a low-ranked anchor would consume one of the four parent slots and could *lose* recall; the fix is that the anchor supplies only the cutoff value and never reorders, which restores the strict-superset property.

Rather than build any of it, the mechanism was simulated offline. A first measurement established headroom (how often the ratio actually binds), then a simulation computed, for each question, what anchor-filter *would* have selected. On `corpus.json` it fixed 1 of 25 (id=25, 0/2 golds → 2/2) and regressed none. Both reranker corpora — formal and abbreviated — came back 26/26 unchanged, falsifying the hypothesis that casual phrasing would make wrong-degree top-1 more common: abbreviation collapses scores dramatically (top-1 average drop 0.44, extremes 0.9883 → 0.0011) while leaving the *ranking* intact. The user then authored `corpus_cross_references.json`: 8 multi-hop questions sampled from the citation graph itself rather than reverse-engineered from the mechanism, split into treatment (the three remaining tight cross-degree edges), weak treatment, and same-chapter controls, with the pass rule fixed in advance. Treatment scored 0/3 — in 7 of the 8 questions the top-1 was already in the requested degree, so the mechanism could not even activate.

The same corpus, however, showed that 6 of 8 multi-hop questions do miss a gold article, and the diagnosis was that the ratio cuts the second article, not that the degree is confused. A ratio sweep was then run once per question with the full scored list persisted, so every subsequent sweep is arithmetic on a stored file rather than an API call. That established the knee, ruled out raising the child cap, and ruled out elbow detection and sub-query count as adaptive alternatives. Finally the real eval was run — retrieval at 0.60, 0.45 and 0.30 with a `--ratio` override added so sweeping never requires editing production config, then e2e generation and scoring at 0.60 and 0.45 on the same day with the same judge.

## Result / outcome  (2026-08-13)

`RERANK_RATIO` is now 0.45. On `corpus.json` this moves `context_recall` 0.9400 → 1.0000 and `context_precision` 0.9433 → 0.9667, with id=17 and id=25 — the two questions the branch was opened to fix — recovering and no question regressing. End-to-end, faithfulness moves 0.8095 → 0.8639 and answer_correctness 0.6797 → 0.7118, though the per-question split is near even and that half of the evidence is inside the noise band for n=25.

The definition-of-done from the handoff ("degree flags are consumed by retrieval, and id=25 / id=17 recover recall while id=23 / id=24 do not regress") is met in outcome but **not** in mechanism. The degree flags are not consumed and will not be: the filter they were built for was tested against 93 questions and changed exactly one, which the ratio change fixes on its own. The ingestion modules and their metadata remain committed as an inert record — zero runtime cost, `qa_chain` reads none of it — so the rejected hypothesis stays inspectable instead of being re-derived from scratch.

The feature also leaves behind reusable evaluation capacity: a multi-hop corpus that closes the "multi-hop underrepresented" gap, a sweep harness whose stored scored-lists make future ratio questions free, and a `--ratio` flag plus self-describing run config so two runs differing only in threshold can never again be told apart by filename alone.

### features/cross_reference/log.md  (feature narrative, durable)

The citation-agent experiment returned to the unresolved multi-hop tail rather than reopening degree filtering. The intent was to retain single-pass retrieval as seed, then let a local gate select individual cited articles. State and observation were separated so the model received compact citation evidence while downstream retrieval consumers retained complete article text. The work was committed in seven groups ending at 7575bbe on harness/cross-reference (2026-09-07), covering schema, tools, local serving/client, gate/prompt, traversal, tests and result snapshots. The production chain and API were not changed by that commit range.

Review concentrated on proving mechanical behavior independently of model behavior: seeds survive verbatim, unselected siblings remain eligible, membership blocks revisits, failed fetches do not enter state, and a supplied stop ends traversal. The suite gained a separate conftest boundary, controlled fake executables, strict-schema cases and final-frontier logging cases. Initial heading failures exposed an overly permissive recognition rule; the committed source now narrows it, and the few-shot lookup now checks matching input/decision pairs instead of depending on example order. These are source-review observations; no pytest or live evaluation was run during this extraction.

**Result / outcome, artifact audit at 7575bbe:** the committed files are `evals/v2/results/eval_retrieval_crossref_baseline_20260906.json` and `evals/v2/results/eval_retrieval_crossref_agent_20260906.json`. Their created_at timestamps are 2026-09-07 04:26:50 UTC and 04:30:54 UTC despite the filename date. Values below were checked directly from the JSON, not inferred from filenames or commit subjects.

| Measurement on 8 questions | Baseline | Agent |
|---|---:|---:|
| Complete gold retrieval | 2/8 | 6/8 |
| Mean gold article recall | 0.625 | 0.875 |
| Mean gold article precision | 0.5417 | 0.475 |
| Mean per-question gold article F1 | 0.5708 | 0.5970 |
| RAGAS context recall | 0.7708 | 0.9375 |
| RAGAS context precision | 0.9375 | 0.9167 |
| Mean context count | 2.375 | 3.875 |
| Mean full-context characters | 3664.875 | 6505.125 |

Recovery applied to six incomplete baseline rows and recovered all missing gold in four (IDs 3,4,5,7). IDs 1 and 6 retained complete gold but added three and two articles respectively. IDs 2 and 8 had seeds {5,13} and {30,40}, missing sources 22 and 42; neither invoked the gate. All eight recorded agent seed ID sets match their corresponding baseline retrieved-ID sets. This supports attribution of ID recovery to follow fetches for these artifacts; it is not a guarantee that two fresh retrieval runs will share seeds.

The trajectory contains 12 gate calls and 12 successful follows, zero stop decisions, six no_candidates terminations and two follow_limit_reached terminations. Four added articles were new gold and eight were outside the labeled gold sets. Eleven decisions selected the first candidate; the exception was id 5 call 3 choosing 8 from [40,8,43]. Seven fetches occurred after gold was already complete: id 1 (three), id 3 (one), id 5 (one), id 6 (two). The local handoff comparison omitted id 3 in its explanatory enumeration and called the id 5 choice index 3; the JSON shows the second candidate, zero-based index 1. Its approximate +54% character estimate is also superseded here: the saved full texts give +77.5%. This corrects the disposable report without treating it as authoritative over the artifacts.

The measured outcome is improved complete-gold retrieval with reduced set precision and a small F1 change. It demonstrates that citation fetches can recover required internal articles in this sample. It does not establish that the LLM adds value over deterministic expansion or that generated answers improve. The reproducibility and validation gaps discovered at the committed checkpoint are tracked in tracker.md rather than silently treating the stored results as a reproducible end-to-end release.


## 2026-09-09: Three initial-selection implementations reviewed

Reviewed working files in `/home/coronny/rag-project` (feat/exp-harness-citation), its `codex-initial-selection` child (codex/initial-selection), and its `initial-selection` child (initial-selection). The v2 harness exists and the user reports reproduced results; the previous checkpoint's missing-harness debt is historical, not a current instruction to rebuild v2.

All three use baseline JSON import, frozen seeds, existing frontier/observation helpers, first/LLM policies, separate result artifacts, and deterministic metrics. Compared loop/tools/gate/prompt/client/config file hashes were identical across the three checkouts. No implementation changes or live LLM calls were made during review.

Fresh isolated suites passed: main 91, codex child 33, initial-selection child 27. Command per worktree: `PYTHONDONTWRITEBYTECODE=1 /home/coronny/rag-project/.venv/bin/python -m pytest evals/agent-exp/tests -q --confcutdir=evals/agent-exp/tests -p no:cacheprovider`. Synthetic probes additionally reproduced:

- initial-selection: STOP on a follow label raises ResultRecord ValidationError rather than saving an incorrect decision (contracts.py:459; metrics.py:129)
- codex/initial-selection: different_trial_ids uses symmetric difference of result keys, so a valid pair choosing different articles reports agreement 0 but an empty disagreement list (metrics.py:202)
- codex/initial-selection: interruption on trial two leaves one persisted row but summary valid=0/missing=2 because results are merged only after an entire policy finishes (run_experiments.py:357)
- main and codex child: changing persisted correctness flags to true for an incorrect decision is accepted and yields accuracy 1; summary does not revalidate those flags against the decision
- main and codex child: a case with nonempty candidates can be relabeled no_candidates and excluded without rejection
- main and initial-selection child: duplicating the same question under another case_id is accepted as a second eligible case
- main: policy-error run reports completed_with_errors but exits 0 (run_experiments.py:392)

Additional source findings: main/codex save seed_order=[] while initial-selection records actual context indices; main cost aggregation excludes error latency; codex by_position includes single-candidate follows and excludes STOP, while initial-selection uses valid multi-candidate outputs including STOP. These supplementary distributions are not directly comparable. Initial-selection reloads persisted results before finalizing; codex buffers the current policy; main tracks each result in memory. Codex has the strongest complete case-roster checks; initial-selection has stronger result invariants but the STOP invariant is defective.

Recommendation delivered: initial-selection is a reasonable foundation after fixing its STOP and duplicate-question defects, borrowing codex's case-roster validation. No merge or branch choice has been authorized. All findings remain unfixed at this review snapshot. Detailed ephemeral resume evidence is in `.knowledge/handoffs/2026-09-09-agent-exp-three-worktree-review.md`.

### features/cross_reference/log.md  (feature narrative, durable)

The initial-selection milestone turned the earlier traversal concern into a paired decision experiment. The user finished labels against frozen seed-derived cases, ran the original-order comparison, and committed the experiment in five groups ending at 2dfa92c on feat/exp-harness-citation (2026-09-10). The checkpoint includes the harness, isolated tests, frozen inputs, measured results and setup guide. Its purpose was to establish a baseline for explaining gate behavior before changing the prompt or measuring a full retrieval trajectory.

The artifact reviewed was `evals/agent-exp/results/20260910T040325Z_initial-selection_0fcb68f9912f/`, retained in result commit 2484b78. Each policy produced six valid decisions, with no errors or missing trials; questions 2 and 8 were mechanically excluded for empty frontiers. The policies agreed on all six decisions: correct follows for questions 3, 4, 5 and 7, and incorrect follows on stop-labeled questions 1 and 6. Both therefore scored Selection Accuracy 4/4 and Decision Accuracy 4/6; first-position selection was 4/4 among multi-candidate trials. The LLM added no measured decision advantage in this run. These paired outcomes motivated the next ordering experiment rather than a prompt change.

Source review at the checkpoint confirmed that the two missing policy-error summary assertions identified in the remaining-refinements review are now present. This extraction checked the committed anchor, source and saved measurement artifacts; it did not execute pytest or rerun the model. The historical 89-pass consolidation result is prior offline evidence, not a fresh test result for this checkpoint. Initial-selection reached the limited measurement objective: human-labeled, paired gate decisions can be examined independently of downstream retrieval outcomes.
