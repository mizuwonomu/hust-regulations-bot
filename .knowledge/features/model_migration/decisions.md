# model_migration — Decisions

## Chosen approach + why

Switch the answer-generation LLM from `qwen/qwen3-32b` to `openai/gpt-oss-120b` (commit 506afcc, branch feature/fastapi, 2026-08-07, on top of baseline a20ebb6). The measurement that backs it shipped just before, as the model-parameterized harness (f1755b4) and its result files (5fa4c6d).

**This was not a choice between two options.** `qwen/qwen3-32b` was decommissioned on Groq and can no longer serve requests, so "keep qwen3" did not exist as an alternative. The measurement below was therefore not run to *select* a model — it was run to check that a forced migration did not silently make the system worse. Framing it as "we evaluated and picked the better model" would be a reconstruction of history that never happened.

The evidence, on 25 questions of `evals/datasets/corpus.json` with retrieval frozen to `eval_20260405_063409.json` (identical context for both models, so the generator was the only variable):

- answer_correctness 0.6315 → 0.6951 (+0.0636); 14 questions improved, 6 regressed beyond ±0.05. This is a real directional signal.
- faithfulness 0.7753 → 0.8108 (+0.0355); 10 improved, 7 regressed, 8 unchanged. Per-question swings reach ±0.5, so a mean shift of 0.0355 on n=25 is **inside the noise** — this is "not worse", not "better". It must not be reported as a faithfulness improvement.
- id=19 (the known table-comprehension failure) flipped to correct (faith 0.75→1.0, corr 0.231→0.611). Encouraging, but it is one question in a set where 7 others regressed on faithfulness; it carries no more evidential weight than id=14, which lost 0.5.

## Assumptions it rests on

- **The comparison is generation-only.** Both runs consumed the same frozen retrieval file, so the numbers describe the answer-generation layer, not the end-to-end system. They are *not* the bot's quality figures, and they say nothing about `RERANK_RATIO = 0.6`, which was never active in the retrieval that produced this context.
- **The 25-question corpus is representative enough to detect a large regression, not a small one.** With per-question variance of ±0.5, only gross degradation would have been visible. A subtle, systematic weakness in gpt-oss (a phrasing style, a citation habit, a Vietnamese register) would pass this test undetected.
- **The two runs are comparable across time.** The qwen3 baseline was scored on 2026-06-27; gpt-oss on 2026-08-07. Same judge model and same corpus, but a RAGAS version change or judge drift between those dates would silently shift the scale. This assumption is now permanently unverifiable.
- **`reasoning_format="parsed"` behaves equivalently for both models.** The user manually inspected the generated responses and confirmed no chain-of-thought leaked into `predicted_response`, so the judge scored comparable artifacts. This was checked, not assumed.

## Failed approaches

- Tried: **Re-running the qwen3 baseline on the current code so both arms are measured the same day** — Failed because: qwen3-32b is decommissioned and returns no responses; the baseline is frozen at its 2026-06-27 file forever. Avoid when: tempted to "just re-measure the old model" to close a methodology gap — with a retired model that option is gone, and the gap has to be documented instead of fixed.
- Tried: **Attributing the delta to gpt-oss having more parameters** (the user's initial hypothesis) — Failed because: parameter count is not the only difference between the two models (architecture, training data, post-training, Vietnamese capability all differ), so causal attribution to one variable is unsupported by two data points. Avoid when: a bigger model scores higher and the size difference is the most visible explanation — visibility is not evidence.
- Tried: **Reading the mean deltas as the headline result** — Failed because: the faithfulness mean is the residual of two large, nearly cancelling tails (+0.511 on id=25 against −0.500 on id=14), not a lift in the baseline. Avoid when: reporting an A/B where per-question variance exceeds the mean shift — report the win/loss split and the tails, not the average alone.

## Nuances agreed with the user

- **The regression list is the durable artifact, not the averages.** Because qwen3 can never be run again, the per-question losses are the only thing that can ever answer "did the model swap cause this?" if a complaint arrives later. Faithfulness regressions: id=14 (−0.500), id=13 (−0.333), id=8 (−0.250), id=23 (−0.250), id=22 (−0.167), id=6 (−0.143), id=18 (−0.125). Correctness regressions: id=17 (−0.451), id=1 (−0.259), id=12 (−0.233), id=6 (−0.182), id=16 (−0.130), id=15 (−0.107). id=6 is the only question that regressed on both metrics.
- **The commit message must state deprecation as the reason.** An operational constraint is a legitimate and strong reason; hiding it behind the metrics would leave a future reader unable to understand why the model changed at that particular moment, and would imply the choice is freely reversible when it is not.
- **The model label belongs to the result file, not to the operator's memory** (harness reworked in f1755b4). Before this work, nothing in a result file recorded which model produced it — runs were distinguishable only by filename timestamp. That is why the harness now writes `model`/`temperature`/`retrieval_results_path` into run-level metadata, and why the scoring script reads the label from the generated file rather than from a CLI flag: a hand-typed label can disagree with reality, and a wrong label would follow the scores permanently.
- **Comparison stays out of the scoring script.** `run_evals_e2e.py` scores one file and carries the label through; it does not interpret it. Joining two scored files by id is analysis, done inline, and only becomes a script if it is needed a third time.

---

## 2026-08-17 — Second Groq decommissioning (llama family retired)

Discovered while testing G3: Groq retired **both** `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`, so the entire live chain (router / query rewrite / chitchat) plus title generation was calling dead models — the app was broken end to end, not just eval. This is the second unannounced decommissioning, and it validates the "model id = live external dependency" pointer in `index.md`. Only three models survive: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, and a qwen3 (~27b — exact id must be copied from Groq's live list).

**Dead-model map (what broke):** `src/rag/config.py` `ROUTER_MODEL` (8b) / `QUERY_REWRITE_MODEL` (70b) / `CHITCHAT_MODEL` (70b); `src/services/title_generator.py` hardcoded `llama-3.1-8b-instant` (outside config — 17b247c's centralization had missed it). `INFERENCE_MODEL` (gpt-oss-120b) survived. Eval scripts also hardcode dead judge/rewrite ids but were deferred (see below).

**Chosen approach + why** (live-chain swap shipped as commit c740dc7, branch api-migration). qwen3 with `reasoning_effort="none"` is the runtime workhorse for every latency/cost-sensitive path (router, query rewrite, chitchat, title); gpt-oss-120b stays the answer generator; qwen3 with reasoning ON is reserved as the RAGAS judge. The deciding fact — verified against Groq's reasoning docs and Context7 (`langchain-groq`) — is that **gpt-oss models cannot turn reasoning off**: their `reasoning_effort` accepts only `low`/`medium`/`high`, and `include_reasoning:false` merely hides reasoning from the response while the model still generates and bills those tokens. Only qwen3 supports `reasoning_effort="none"` for a true off switch. So on trivial runtime tasks (binary classification, query reformulation, chitchat, an 8-word title) qwen3-none is strictly cheaper and faster, while gpt-oss-20b would force reasoning cost on every call for no quality gain. This **reversed the initial lean toward gpt-oss-20b for the runtime path**, and let gpt-oss-20b drop out of the architecture entirely.

**Assumptions it rests on.**
- The disable value is the **string** `"none"`, not Python `None`. Passing `None` drops the kwarg → Groq default → thinking back on. `ChatGroq` forwards `reasoning_effort` to Groq.
- The qwen3 id in config (`qwen/qwen3.6-27b`) resolves on Groq's live list — copy the exact id from there; a wrong id fails the same silent way the llama ids just did.
- Runtime tasks genuinely don't benefit from reasoning. If rewrite quality drops on multi-hop questions, this is the assumption to revisit — the recall tail is already fragile (see cross_reference).
- **Judge must not equal generator.** Reusing gpt-oss-120b as both generator and RAGAS judge introduces self-preference bias (a model rates its own outputs higher). The judge stays a different surviving model (qwen3).
- The judge path goes through RAGAS `llm_factory` (OpenAI-compat), which has **no** `reasoning_format="parsed"`. qwen3 as judge may leak `<think>` into the content field and break RAGAS parsing. Unverified — must be checked on the first eval run, and the judge wants reasoning ON so it cannot simply be set to `none`.

**Failed approaches.**
- Tried: **gpt-oss-20b + `reasoning_effort="low"` for the runtime path** (the pre-swap plan). Failed because: gpt-oss cannot disable reasoning; `"low"` still emits reasoning tokens on every call, and `include_reasoning:false` suppresses output but not compute or billing. Avoid when: tempted to pick gpt-oss-20b as a "small fast" runtime model — it is not thinking-free, so it is the wrong tool for classification/rewrite/chitchat/title.
- Tried: **`reasoning_effort=None` (Python None) to disable thinking.** Failed because: None drops the kwarg → default → reasoning stays on. Avoid when: assuming "none means None" — Groq expects the literal string `"none"`.

**Nuances agreed with the user.**
- **Runtime unified on one model id, reasoning varies per call site** (`none` for runtime, `default` for judge). One qwen3 id, two modes — clean on latency, cost, and the judge≠generator rule at once.
- **Title-gen pulled into config** (`TITLE_GENERATOR_MODEL`), closing the hardcode that sat outside config.
- **Eval scripts deferred deliberately** (job 3). They stay on dead ids and will fail if run until migrated; the judge≠generator choice and the baseline-epoch break travel with that later work. `JUDGE_MODEL` sits in config as a waiting constant with no live consumer yet.
- **Eval baseline is now a dead epoch.** The judge that scored every prior baseline (`llama-3.3-70b`) is gone, so no old run can be re-judged with the same judge; every cross-run comparison against pre-2026-08-17 baselines is invalid.
- **Migration verified in-process, without the backend.** LangSmith showed router 260 tok / rewrite 546 tok with no reasoning blocks, and gpt-oss still producing inference — confirming `reasoning_effort="none"` takes effect and the generator is intact. Answer + chitchat run inside Streamlit (`get_chain`) with its cached sync connection, so only title generation needs uvicorn to verify.

---

## 2026-08-29 — Job 3: eval revival (branch fix/evals)

Migrated the live-path eval scripts (`run_evals_retrieval.py`, `run_evals_e2e.py`) off the dead llama ids. `run_generate_e2e_responses.py` needed no real work — the generator (`gpt-oss-120b`) is alive and its `_gen_` output files already exist — so job 3's substance was the **judge**, not an id swap.

**Chosen approach + why.** The judge uses `LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, reasoning_format="parsed"))` on the **legacy** `ragas.metrics`, not `llm_factory` on `ragas.metrics.collections`. Rewrite is `ChatGroq(QUERY_REWRITE_MODEL, reasoning_effort="none")`. Both read ids from `src/rag/config.py`. `ragas` pinned `==0.4.3` in `pyproject.toml` + `uv.lock`.

**The deciding constraint (verified against ragas 0.4.3 source).** A qwen3 judge must keep reasoning ON to score, so its `<think>` must be kept out of the field RAGAS parses. Two facts settle the design:
- `llm_factory` requires a native SDK client + Instructor and has **no** `reasoning_format` — the OpenAI-compat path cannot separate reasoning, so qwen3 would leak `<think>` into `content` and break parsing. This is the constraint that originally forced the openai-client path.
- `ragas.metrics.collections` (the non-deprecated API) explicitly **rejects** `LangchainLLMWrapper` (`llm: InstructorBaseRagasLLM`, "legacy wrappers are rejected"). So the only way to use `ChatGroq(reasoning_format="parsed")` — which does split reasoning cleanly — is the **legacy** `ragas.metrics`, whose `Faithfulness`/`AnswerCorrectness`/`ContextPrecision`/`ContextRecall` accept any `BaseRagasLLM` and take `cache` on the wrapper too (no loss of the DiskCache).

So migrating the API *forward* to collections would **reopen** the `<think>` problem. The reproducibility fix for "legacy gets removed at v1.0" is therefore a **version pin (`==0.4.3`), not an API migration** — pinning makes the removal a non-event until we deliberately cross it. A future collections migration is its own task and must solve judge-reasoning-on-Instructor then (extra_body `reasoning_format`, or a groq-native instructor client).

**Verified.** 1-item retrieval probe: `context_recall=1.0`, `context_precision=0.7499999999625` — finite, so RAGAS parsed the per-statement verdicts and no `<think>` reached `content`. The tracker's UNVERIFIED note is closed.

**Nuances agreed with the user.**
- **Legacy metric API differs from collections**: call is `single_turn_ascore(SingleTurnSample(...))` returning a float, not `ascore(**kwargs)` returning an object with `.value`. One `SingleTurnSample` carrying all fields feeds both metrics; each picks what it needs.
- **Two stacked epoch breaks now.** Judge (llama→qwen3) AND metric impl (collections→legacy) both shift scores, so results from this branch are a fresh baseline — never plotted on the same axis as any pre-2026-08-17 run. Doing both at once means a delta cannot be attributed to one; the README must present each number self-describing (judge / metric / ragas version / date), the same discipline the result files already follow.
- **Old scored files are archived, not deleted** (planned): `_gen_` files survive the epoch break (generation is judge-independent, reusable as judge input); `_scored_` files are llama-epoch and go to `results/archive/`. Nothing hard-deleted — git keeps history anyway, `archive/` is for human discoverability.
- **Deliberately still deferred**: `run_ratio_sweep.py` and `run_rewrite_rerank_calibration.py` keep dead ids (frozen snapshots reproducing committed result files — migrating them breaks that reproducibility).

**Still open at time of writing.** The full run (`corpus.json` 25 + `corpus_cross_references.json` 8) has not been executed; only the 1-item probe. Archive + README refresh follow the full run.

---

## 2026-08-30 — Job 3 full run: two runtime bugs, and the epoch-new baseline

The migrated scripts ran end-to-end on `corpus.json` (25) — retrieval → generate (gpt-oss-120b) → e2e score — producing the first qwen3-judge baseline (fix commit 578a235, readme 696702f, on master). `corpus_cross_references.json` was skipped (it lives on an abandoned branch). Two bugs surfaced only at full scale, invisible in the 1-item probe:

**Bug 1 — judge truncation, hidden by the DiskCache.** The qwen3 judge raised `LLMDidNotFinishException` on heavy prompts. Cause: reasoning tokens count toward `max_tokens`, and with no explicit cap Groq's default truncated the verdict (`finish_reason="length"`), which RAGAS's `is_finished` rejects — its allowed set is only `stop`/`STOP`/`MAX_TOKENS`/`eos_token`, and OpenAI/Groq's `"length"` is NOT in it. Fix: `max_tokens=10000` on both judges. The trap that cost two failed runs: RAGAS caches the raw LLM response *before* the is_finished check, so run 1 (small cap) cached a truncated response and later runs read that cached truncation and failed identically — the new `max_tokens` never fired until the cache was cleared. **Lesson (cross-cutting for any cached-LLM work): changing an LLM param while a DiskCache is warm tests new code on old answers — clear the cache.** A heavy 4-context precision prompt measured ~2189 reasoning tokens, so 10000 is generous; the real driver is variance (rewrite runs at `temperature=0.2`, so per-run context length — hence judge reasoning length — differs), which is exactly why the 1-item probe passed and the full run failed.

**Bug 2 — AnswerCorrectness sub-metric never initialised.** Legacy `AnswerCorrectness` builds its `AnswerSimilarity` (the semantic half) inside `.init(run_config)`, which RAGAS's `evaluate()` calls but a direct `single_turn_ascore()` does not — so it asserted "AnswerSimilarity must be set". Fix: construct `AnswerSimilarity(embeddings=...)` explicitly and pass it in. Faithfulness / ContextPrecision / ContextRecall have no sub-metric, so only AnswerCorrectness needed this.

**Baseline (epoch: qwen3 judge, legacy metrics, ragas 0.4.3, gpt-oss-120b generator, ratio 0.45, 2026-08-30):** context_recall 0.9567, context_precision 0.8933, faithfulness 0.8515, answer_correctness 0.8446 (n=25, no NaN).

**Nuance — is qwen3-27b judging differently from the retired llama-70b?** Directly unmeasurable (the old judge is dead; old files used different retrieval — the epoch break). What IS checkable is *defensibility*: on id=22 (precision 0.25) every per-doc verdict was inspected and correct — the judge rejected three wrong-degree registration articles and accepted the one gold (Điều 19), so the low precision is a retrieval-ranking failure (gold ranked last), not judge harshness. `context_precision` also penalises rank, so a perfect judge still returns 0.25 when the single relevant doc sits at position 4. Where 27b-vs-80b divergence could hide is *borderline* docs, not clear-cut ones — that stays unmeasured. Cheap proxy if it ever matters: A/B the same contexts against gpt-oss-20b (alive, different family).

**README deliberately NOT given the new numbers.** The v1/v2 table in `docs/README.md` was marked "(archived results)" with an italic "older LLM judge, now archived" note rather than replacing its figures — the new baseline is a fresh epoch that confirms nothing against the old, so showing a delta would be dishonest.
