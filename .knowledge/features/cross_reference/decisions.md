# cross_reference — Decisions

## Chosen approach + why

**Lower `RERANK_RATIO` from 0.6 to 0.45** (commit d7611fd, branch fix/cross-reference, 2026-08-13) — and ship the degree-metadata work as an inert checkpoint only. The feature set out to fix wrong-degree-level multi-hop retrieval by consuming the regulation's citation graph; it ended by **falsifying that hypothesis** and finding the failure was a threshold calibration problem instead.

- The knee moved because **0.6 was calibrated on an all-single-hop corpus**. A single-hop question is answered by one article, so a tight ratio is free. A multi-hop question needs a second article joined by *citation*, not by wording — the reranker scores it far lower by design, and 0.6 cuts exactly there.
- Measured on `corpus.json` at identical code/date/judge: `context_recall` 0.9400 → 1.0000, `context_precision` 0.9433 → 0.9667, e2e `faithfulness` 0.8095 → 0.8639, `answer_correctness` 0.6797 → 0.7118. id=17 and id=25 recover; no question regresses. Artifacts `eval_ratio{060,045,030}_20260813.json`, `eval_e2e_scored_ratio{060,045}_20260813.json` (commit 7ff8f10).
- **Precision went up, not down.** The whole session had assumed lowering the ratio buys recall with precision. It does not: RAGAS `context_precision` asks whether retrieved context is *usable*, and at 0.6 id=25 retrieved one wrong context scoring 0.0000. Adding correct parents raised that question's precision to 0.5833. A "count the non-gold parents" proxy predicts the opposite and is wrong — do not reason about precision by counting documents.
- **0.45 over 0.30**: both give an identical aggregate (1.0000 / 0.9667); 0.30 only pulls in more context, so the tighter value wins.
- The degree-flag ingestion (commits d5c7a7b, 654509d, bb044ee) is kept **stored but unconsumed**: `qa_chain` reads none of it, `page_content` is untouched so vectors are unchanged, and the runtime cost is zero. It is a checkpoint of a rejected hypothesis, not a live feature.

## Assumptions it rests on

- **Single-hop recall is insensitive to the ratio.** Measured across `corpus_reranker` (formal) and `corpus_reranker_abbre`: recall stays 1.000 / 0.963 at *every* ratio from 0.60 down to 0.05. This is what makes 0.45 free rather than a trade. If a future corpus contains single-hop questions where a wrong article sits just under the old cutoff, this breaks and the knee must be re-derived.
- **The answer LLM tolerates extra context.** 0.45 admits ~0.4 more children per question. This is only safe while the generator obeys the "answer only from context" prompt; the e2e run confirms it for `openai/gpt-oss-120b` at temperature 0.1 and for nothing else.
- **The multi-hop evidence is 8 hand-written questions plus 2 in `corpus.json`.** The knee is directional, not a proven optimum, and rests on a corpus the same session authored.
- **RAGAS judge stability.** Both arms of the 0.60-vs-0.45 comparison were scored on the same day by the same judge model. Comparing either number against a run from another date is invalid.

## Failed approaches

- Tried: **Degree filtering — filter candidates by `ap_dung_*` before rerank** — Failed because: 7/8 of the purpose-built multi-hop questions already have a top-1 that is *in* the requested degree, so the mechanism never activates; across 93 questions it changed exactly one, and that one (`corpus.json` id=25) is fixed by ratio 0.5 alone. The prerequisite frontend (degree picker, session state, request plumbing) would have been built for a single question. Avoid when: a failure looks like "wrong degree level" — verify the wrong-degree document is actually *outranking* gold before building anything, because scoring low and being the wrong degree are different diseases. **Cross-cutting** (would have touched child metadata schema, retrieval, and frontend state) — pointer promoted to index.md.
- Tried: **Anchor-filter — let the first in-degree child define the ratio anchor instead of top-1** — Failed because: it is dominated by the plain ratio change. It fires only when top-1 is out-of-degree (4/25 on `corpus.json`, 1/8 on the multi-hop set), and its effect size is proportional to `1 − anchor/top1`, which is 0.93–0.95 in most firing cases, so the cutoff barely moves. Avoid when: tempted to make the selection rule degree-aware "since the metadata is already there" — the metadata being free does not make the mechanism useful.
- Tried: **Chroma `where` clause for degree filtering** (rejected at design time) — Failed because: retrieval is hybrid; `BM25Retriever` is built from documents in memory and has no metadata filter, so a `where` clause filters only the dense arm and leaves the pool half-filtered. It also changes `k` semantics (dense returns k *post-filter*), invalidating any earlier baseline. Avoid when: adding any metadata gate to retrieval — it must sit after the merge, never inside one arm.
- Tried: **Adaptive cutoff by elbow/gap detection** (largest relative score drop) — Failed because: it is strictly dominated by the constant. At equal noise on `corpus.json` it scores 0.812 recall against fixed-0.30's 0.906. The reason is structural: for multi-hop the second gold article is *supposed* to score far below top-1, so the largest gap frequently sits exactly at the boundary the query needs to cross. Avoid when: reaching for a "smarter than a magic number" threshold — this is the same class of error as the hard floor, reading meaning out of the score column's shape. **Cross-cutting** (score-comparability) — folded into the existing index.md pointer.
- Tried: **Sub-query count as a multi-hop signal** (to drive an adaptive ratio) — Failed because: the query-expansion LLM emits exactly 3 sub-queries for all 85 questions regardless of complexity; the prompt's "maximum 3" behaves as "always 3", so the signal carries zero information. Avoid when: looking for a free complexity signal already in the pipeline — measure its variance before designing on top of it.
- Tried: **Raising `RERANK_MAX_CHILDREN` above 5** — Failed because: the cutoff binds long before the cap does (average kept is 2.5 children at ratio 0.6), so raising the cap yields +0.000 recall at every practical ratio and +0.010 only at ratio 0.2 with cap 10. Avoid when: a gold article is seen sitting at rank 6+ — it needs a lower ratio first, and by then the cap is worth only about one gold across the whole corpus.
- Tried: **Forward citation-graph expansion** (pull cited articles into context after selection) — Failed because: simulated offline against all 7 multi-hop questions it fixed zero failing cases and added noise (id=10 gains Điều 3, not gold); id=25's post-ratio seed is `[17]`, which has no outgoing edges, so both golds were cut before expansion could reach them. Avoid when: the citation graph looks like an obvious recall fix — expansion cannot recover what selection already discarded.
- Tried: **Reverse citation-graph expansion** (pull citing articles) — Failed because: fan-in is too broad — Điều 3 is cited by 6 articles across all four degree levels, so id=1 would gain 7 extra articles and id=21 five, while rescuing only id=17. Avoid when: forward expansion fails and reversing the edge direction looks like the natural next attempt.
- Tried: **pyvi word segmentation, NFC normalisation, RRF→union in the retrieval merge** (measured in the prior session) — Failed because: segmentation improves rank but not pool membership and the pipeline discards BM25 order since fab3e07; NFC is a no-op (the corpus is already NFC, 0/170 children changed); RRF→union produces the same set and the order is unread downstream. Avoid when: hunting recall in the retrieval merge — these three are already ruled out.

## Nuances agreed with the user

- **High faithfulness can be a symptom of broken retrieval.** At ratio 0.6, id=25 retrieved nothing correct, so the model returned the deterministic refusal — which scores 1.000 faithfulness because it asserts nothing. Moving to 0.45 dropped that question to 0.875 faithfulness while correctness went 0.098 → 0.755. Never read faithfulness without correctness beside it.
- **An aggregate gain is not a broad gain.** The e2e improvement is carried by a few large swings: per-question the split is 6 win / 5 lose on faithfulness and 13 win / 12 lose on correctness. On n=25 with ±0.5 swings this is inside the noise, exactly as `tracker.md` already warns for the model migration. The retrieval numbers are the more trustworthy half of the evidence because they do not depend on an LLM judging free text.
- **Ratio is a score threshold, not a document quantile.** `0.45` means `cutoff = top1_score × 0.45`; how many documents survive is decided entirely by the shape of the score column. Pools of 30–56 candidates typically keep 2–4 children, i.e. under 10% — not 45%.
- **The eval-vs-production drift class recurred and was caught.** `eval_retrieval_ratio_20260708.json` proved unreproducible: it carries an "Điều-only" context prefix produced by uncommitted local edits, while both the committed code (bare `page_content`) and the fix (`Chương - Điều`) differ from it. It was therefore refused as the 0.6 baseline and a fresh 0.6 arm was run instead. Every measurement script in this feature was placed in `evals/v2/scripts/` for that reason, never in a scratch directory.
- **Degree scope is a set, not a scalar, and flags may only be turned ON.** Chương gives the base (II=ĐH, III=KS, IV=ThS, V=TS; I and VI apply to all), and `dependency` citation edges extend a target with the borrower's level — Điều 10/12/13/20 live in the ĐH chapter but are borrowed by KS. The graph is a lower bound, so the flags are conservative by construction.
- **Flags are Điều-level, not khoản-level**, because children are cut by character length and straddle khoản boundaries. The known cost is Điều 12 khoản 5 inheriting an undeserved KS flag. Chroma metadata also rejects lists and has no substring predicate, which is why the scope is four booleans rather than one `"DH|KS"` field.
- **A child carrying no degree flags is treated as matching every degree.** Exactly 1 of 170 children is in this state (the document preamble, before the first Chương header). Treating a metadata gap as an exclusion would turn an ingestion hole into a silent retrieval bug.
- **Intake year is a second fragmentation dimension** with the same shape as degree — Điều 48 scopes khoản by intake cohort, the citation edge carries no year, and nothing knows the asker's cohort. This is a missing input, not a retrieval defect.
- **Two errors were found in the user's hand-built citation analysis**: Điều 17 never cites Điều 8, and the Chương VI count differs only because one citation names three khoản of Điều 17.

### decisions.md  (why — the expensive part)

Citation-agent checkpoint (commits bc3711e through 7575bbe, branch harness/cross-reference, 2026-09-07):

**Chosen approach + why**

- Keep citation traversal outside the production answer chain while establishing a measurable retrieval baseline. This makes recall recovery inspectable without conflating it with generator behavior or changing live chat retrieval. Committing the experiment is a checkpoint, not proof that it should replace production retrieval.
- Keep seed retrieval, article fetch, citation extraction and gate decisions injected. The loop owns traversal state; the model chooses only the next action. This permits deterministic tests of control flow without treating model judgment as a CI invariant.
- Use one insertion-ordered collected mapping for full texts and membership. Unknown or duplicate seed strings must survive independently; unmatched positive metadata IDs remain membership-only entries. Never zip an unordered ID set to texts. This preserves the retrieval seam without inventing source identities.
- Rebuild the forward frontier from all collected texts because unselected siblings must survive alongside citations discovered in newly fetched articles. This does not repeat seed retrieval. The successful-follow cap bounds extra article fetches, not graph depth, gate calls or total seed-plus-follow context count.
- Give the gate identity/title and citation-bearing lines, retaining full articles for downstream consumers. Compact observations focus attention on edges and reduce repeated input, but cannot establish global answer sufficiency. The graph is derived from collected text rather than a second runtime citation index.
- Keep physical GGUF selection in the shell and expose a stable citation-agent API alias to Python. Request-scoped grammar must preserve existing client extra_body settings. Grammar constrains syntax and candidate membership, not relevance or stop quality.

**Assumptions it rests on**

- A missing target must be reachable through forward references in collected text to be recoverable. A missing source is not recoverable merely because its target was retrieved. Article lookup IDs must represent the same corpus and parent identities as seed retrieval.
- Texts use the corpus heading convention and article IDs are meaningful within one regulation. The whitelist is numeric: an external reference with a colliding internal number can still be mistaken for an internal edge. Do not infer document identity from the number alone when expanding to multiple regulations.
- Compact excerpts are lines, not reconstructed semantic paragraphs; page breaks can split a condition from its citation. Increasing few-shot count cannot restore evidence absent from the observation.
- Set metrics assume the labeled gold set defines the required internal articles. The user confirmed gold {22,13} for id 2 and {42,40} for id 8; external presentation guidance is outside the internal corpus gold. Outside-gold articles are not automatically false statements or useless to every answer.

**Failed approaches**

- Tried: interpreting LangSmith Input assistant messages as executed decisions -> Failed because: those messages include fixed few-shot examples, including stop followed by an unrelated example -> Avoid when: diagnosing a loop that allegedly follows after stop; inspect Output and the same query run's gate/fetch events instead.
- Tried: diagnosing ratio failure from ensemble rank 1 -> Failed because: child scores are recomputed against the original question after subquery merge; ensemble order is not reranker order -> Avoid when: a source article appears early in ensemble but is absent from final parents; inspect score, cutoff, child cap and parent fetch in sequence.
- Tried: using stubbed stop tests as proof of model selectivity -> Failed because: these tests only prove the loop obeys a supplied decision -> Avoid when: citing a green mechanical suite as evidence the LLM knows when to stop.
- Tried: running nominally offline agent tests through the shared test configuration -> Failed because: parent conftest loads dotenv and has an autouse database cleanup fixture, causing DB dependency or skips -> Avoid when: running this suite without its confcutdir boundary. This is a cross-cutting test-isolation constraint.
- Tried: leaving system directories on fake-server PATH -> Failed because: the missing-server case could discover a real installation -> Avoid when: testing executable discovery; expose only controlled dependencies and invoke bash by absolute path.
- Tried: asserting grammar RHS strings and selecting the first few-shot with a context ID -> Failed because: harmless formatting or example reordering could break tests, while an unused candidate rule could pass -> Avoid when: testing prompt refactors; check paired example meaning and the root-to-candidate rule relationship.

**Nuances agreed with the user**

- stop means no current candidate edge is needed. no_candidates is a mechanical result, not an error or completeness claim. gate_stop breaks the loop immediately. Only the follow-limit path recomputes a final frontier for logging; this does not request another model decision.
- A follow_limit_reached reason alone says nothing about remaining candidates. Log their final list/count independently; cap 0 must work without referencing an observation that was never built.
- Distinguish gold_article_recall, gold_article_precision and gold_article_f1 from answer correctness. Recall measures required-ID coverage; precision penalizes extra IDs irrespective of rank; F1 balances the two. all_gold_hit is complete coverage, not exact set equality. Recovery applies only to gold missing from the comparison baseline, while added articles must be measured against the agent's actual seed.
- The older statement above that counting non-gold documents is a wrong precision proxy applies to predicting RAGAS precision, not to the separately defined set precision. RAGAS context precision is rank-sensitive and can leave trailing noise unpenalized; its context recall measures supported reference claims rather than rank-weighted article coverage.
- Freeze a baseline before tuning prompt, reasoning settings or ratio. Choosing candidates[0] mechanically is a proposed same-seed control for the LLM's incremental value, not a chosen replacement design or an experiment already performed. The old forward/reverse-expansion failures above belong to their own corpus and setup, not a universal impossibility claim.
- Heading-negative fixtures were robustness cases, not observed malformed corpus headings. The committed matcher now requires a dot after the article number and a constrained chapter prefix; retain this distinction when explaining why old tests failed.


## 2026-09-09: Agent experiment contracts and review lessons

- Keep the new decision experiments in `evals/agent-exp/`, independent of v2 and RAGAS. Reuse the existing eight questions; derive seed snapshots and gate cases from reproduced single-pass baseline output. User approval adds action/acceptable-candidate labels, not a replacement question corpus
- Specs and plans are English documents under `docs/superpowers/specs/` and `docs/superpowers/plans/`. Initial-selection is implemented and reviewed first; permutation and stop-policy follow later. Source-loss/reranking work stays deferred
- Freeze full parent text, order, and separate ID membership after parent selection/fetching. Whitelist is a JSON array of unique positive internal article IDs, supplied as data; do not derive it from gold or seed membership
- First-position Selection Rate and consistency are behavioral metrics that do not require gold labels. Selection/Decision Accuracy require reviewed state-specific labels. Consistency is agreement of observable actions/article IDs, not proof of consistent internal reasoning
- Preserve STOP as a valid output even on follow-required cases: selection_correct=False there, but null on stop-labeled cases. A schema that forbids false selection correctness for all STOP outputs breaks evaluation of false stops
- Persist policy-specific compact results and summaries; keep verbose decisions, trajectories, and logs local/ignored. Independent state replay is not a new trajectory
- Finalization should account for persisted results after interruption. A per-policy in-memory batch that is merged only after the whole policy completes can disagree with rows already written to disk
- Recomputing summary must validate or derive correctness from actual decisions and approved labels, not merely trust serialized correctness flags. Mechanical no_candidates must agree with the frontier, and hop-0 cases must not duplicate a question under different case IDs
- Existing green harness suites do not prove gate semantics or coverage of reporting failure paths. Record source-review observations, synthetic reproductions, offline suites, and live model evaluations separately

## 2026-09-09: Consolidating review tests into the initial-selection harness

- A passing replacement suite does not establish coverage equivalence with the original suite. Map source scenarios to actual assertions, distinguishing input validation, replay validation, artifact loading, and summary validation
- The user limited consolidation to scenarios in review/feat-harness-original/tests and review/remaining-refinements.patch, preserving existing eval tests. Do not introduce new test scenarios merely to broaden coverage. New explanatory comments were specified verbatim by the coordinating agent
- Preserve real loop/gate boundaries with injected synthetic dependencies: capture hop-0 input before fetching, and fake the external client while retaining the real gate parser. These checks do not validate live LLM judgment
- Test actual repository ignore behavior, not a test-created literal; test debug independence by deleting fixture debug and comparing summaries; compare recorded source hashes to source bytes rather than a hash to itself
- Transport classification must recognize concrete HTTP exception types rather than depend solely on message text. Seed import can avoid loading store-related modules by importing the existing loop/tools helpers only when constructing cases
- The old optional num_samples consistency assertion was not adopted: actual baseline/dataset roster alignment remains required. Old approved-unresolved/no_candidates rejection expectations were adapted to the current exclusion contract

### decisions.md  (why — the expensive part)

**Chosen approach + why**

- Retained initial-selection as an independent hop-0 measurement milestone (checkpoint 2dfa92c, branch feat/exp-harness-citation, 2026-09-10) because the question was whether the LLM adds decision value over always following the first candidate. A full loop would mix this question with fetching, later states and retrieval coverage; success meant a trustworthy comparison, not an LLM win
- Kept scripts separate from data and results because the user found code at the experiment root ambiguous. Kept snapshots and labeled datasets together as the frozen input checkpoint because cases refer to the snapshot and later seed-order experiments need its full texts and order, not only hashes
- Preserved the original run manifest after the later commits and directory move because its revision and dirty-tree flag describe execution time. Replacing them with the new HEAD would falsify provenance; stored code hashes remain the evidence for the executed source
- Preferred permutation on the fixed v1 cases before adding a v2 corpus because changing question content and ordering simultaneously would obscure the original position hypothesis. This was the recommended next experiment, not an authorization to run or tune it during extraction

**Assumptions it rests on**

- Seeds are a genuine single-pass baseline from the same corpus, and the whitelist describes internal articles independently of the answer gold set. Revisit seed preparation when corpus identity or the retrieval baseline changes
- Labels judge the next useful citation edge from the actual gate input. An acceptable intermediate article need not equal the final answer's complete gold set; a stop means no remaining edge is needed, not proof that the entire answer is already supported
- Candidate-order comparisons preserve question, observation, candidate membership and model settings. Because candidate order also determines grammar alternatives, their conclusions concern the current list-plus-grammar path rather than an isolated prompt-list effect

**Failed approaches**

- Tried: diagnosing position bias from complete first/LLM agreement on original-order inputs -> Failed because: candidate identity and position were never separated, so semantic selection and first-position selection could yield the same outputs -> Avoid when: interpreting one original-order replay as causal evidence
- Tried: treating the absence of a reasoning trace as evidence that the model merely copied the first candidate -> Failed because: thinking was disabled, the prompt requested JSON without explanation, and debug stored parsed decisions rather than the full raw response -> Avoid when: inferring internal reasoning from this logger's output

**Nuances agreed with the user**

- Observation comes from collected seed titles and citation excerpts, not from the text of candidates[0]. Candidate IDs describe possible next fetches; their full article text has not yet been fetched in this replay
- Retain single-candidate cases for action decisions, but report them separately from position analysis. Repeating or rotating a case adds trials, not independent semantic questions
- Keep the first run and its approved labels fixed when progressing to another experiment. If labels change, create a new case version and comparison rather than silently changing the meaning of old results
- In the remaining-refinements patch review, grouped help/invalid-command tests and the renamed repeat-schedule test preserved the original checks. Error-record persistence alone did not cover summary counts, so the requested refinement was to restore errors=1 and valid=0 assertions in the existing CLI failure test rather than expand unrelated coverage

## 2026-09-10: Permutation review interpretation and aggregation boundaries

- Gold-position accuracy must group by the gold position in each trial's candidate_order, not its position in the original case. Otherwise rotation effects disappear into the original-position bucket even while overall accuracy remains plausible
- Position distributions include valid decisions only. selected_position=None also occurs for errors; it is not sufficient evidence of STOP. Scheduled accuracy still counts errors/missing trials as incorrect
- Consistency group identity must include case_id alongside repeat_id or permutation_id. Flattening case-local groups by their local key overwrites earlier cases and corrupts pooled pair coverage
- Identical full inputs require both observation identity and ordered candidates. In combined experiments, equal observation hashes with different candidate orders are distinct policy inputs
- Candidate-order isolates the current candidate-list/grammar path; seed-order fixes the candidate list while changing context order. Decision changes alone show order sensitivity; claiming preference for the first/last context requires relating chosen citations to their source positions, which can be ambiguous when several seeds cite the same article
- High article-based permutation consistency with low accuracy means stable wrong decisions, not position bias by itself. Repeat consistency measures identical-input stability separately. Full-loop usefulness and internal reasoning cannot be inferred from either
- A no-overwrite test using wall-clock-derived directory names can be flaky across a second boundary. Distinguish path collision protection from the stronger contract that the same manifest always maps to the same directory; this inherited issue must not be blamed on one permutation implementation

### decisions.md  (why — the expensive part)

**Chosen approach + why**

- Closed the fixed-v1 ordering comparison before changing questions or prompts (archive checkpoint fd7696d, branch feat/exp-harness-citation, 2026-09-11) because a stable dataset separates the effect of input ordering from a changed evaluation target. This is an experiment checkpoint, not completion of the broader citation-agent feature
- Used the Codex foundation with selected DeepSeek integration scenarios and compatible per-case reporting inspired by GLM because one authoritative schedule/reconstruction/scoring path is easier to validate than merging three competing implementations. Independent metric counterexamples, rather than donor test counts, determined which formulas to retain
- Used three repetitions per exact input as a small diagnostic budget because one repetition cannot measure stability and two provide only one comparison pair. Three provide three dependent pairs; this was never a statistical sufficiency threshold. Repeat control includes the LLM under the same recorded settings as both ordering treatments
- Kept compact manifests, per-policy decisions and summaries in Git because these three complementary runs form a useful baseline and together added only about 307 KiB of non-debug data at review. Verbose debug remains local; repository growth should be controlled by retaining decision-relevant runs rather than discarding the evidence needed to rescore them
- Kept specifications private at the user's request because implementation and results should be publishable without exposing design documents. Future runner metadata explicitly reports unavailable private-spec provenance instead of requiring a public spec path; setup guides remain the public operating contract. Historical path/hash metadata does not contain spec text and must not be silently rewritten to pretend it was recorded differently

**Assumptions it rests on**

- Ordering comparisons require the same frozen snapshot, labels, prompt and runtime configuration, with candidate membership unchanged. Revisit comparability whenever any of those changes; alias equality alone does not verify the served model weights
- Larger candidate lists must arise from the real retrieval/rerank/fetch-to-frontier path. A source article containing many references is insufficient if deduplication, collected membership or whitelist filtering leaves fewer gate candidates
- The next corpus measures gate selection conditional on usable retrieval states. Rejecting cases whose source is not retrieved is legitimate for this conditional question only if exclusions are retained; it cannot establish end-to-end retrieval quality

**Failed approaches**

- Tried: inferring a fixed-position rule from agreement with first on original-order inputs → Failed because: gold and the selected article were initially first, and candidate rotations later separated article choice from list position → Avoid when: an unpermuted dataset aligns correct targets with candidates[0]
- Tried: interpreting stable choices after seed rotation as proof the model ignores observation → Failed because: seed rotation preserves semantic content, so a content-sensitive policy can correctly remain invariant → Avoid when: only order, not evidence or question meaning, changes
- Tried: treating more candidates alone as a harder semantic test → Failed because: unrelated distractors can leave one obvious answer and repeated questions add no independent coverage → Avoid when: expanding a corpus to meet a numeric quota without reviewing competing citations

**Nuances agreed with the user**

- The accepted conclusion is resistance to a fixed candidate-list position on these observed cases, not proof of internal reasoning or absence of every position effect. Seed-order invariance is also limited to the rotations actually tested; candidate-order includes the current grammar/list coupling
- FOLLOW/STOP causes were explicitly left to the separate stop-policy specification because this experiment does not distinguish corpus bias, prompt few-shot bias and observation/label mismatch. Do not tune the prompt solely to explain these v1 outcomes
- Proposed a practical first expansion of at most 30 candidate questions to retain about 15-20 reviewed questions: roughly 8-10 with three candidates, 5-7 with four or five, and 2-3 with six or more if naturally available. These are provisional curation targets, not requirements or a statistical guarantee; quality and review effort take precedence over filling quotas
- Preserve v1 and create a separate v2. Prefer unfamiliar questions, same-topic distractors and a clear required target; vary source articles and spread useful citations across multiple seeds for seed-order tests. Approve labels before seeing LLM outputs, and require the target to remain outside the frozen collected set
- Questions 2/8 were not diagnosed as internal_dieu bugs. Trace missing source retrieval, reranker/cap removal, extraction, already-collected targets and whitelist filtering before naming a cause; retain the exclusion stage alongside rejected question candidates
- The user chose to run while reviewing uncommitted code, then archive afterward, and accepted loss of uncommitted donor variants to simplify the workspace. A source checkpoint was advice for provenance, not a runtime prerequisite; the integrated root harness had no donor-worktree dependency
