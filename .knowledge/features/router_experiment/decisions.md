# Router Experiment Decisions

## 2026-09-23 - Isolated Jev comparison

- Evaluate Jev against the existing Groq router in `evals/router` before considering any production change. The experiment calls only the routing decision, not retrieval or answer generation
- The evaluation corpus is `evals/router/datasets/corpus_classification.json`: exactly 30 objects, each with exactly four keys `id`, `query`, `type`, `group`; `id` (integer) and `group` (free-form string) are debug annotations only, are never sent to a model, and are never used to sort, group, schedule or join trials; only `query` is a model input and only `type` is a scoring label
- Editorial targets are 18 complete well-spelled queries, 6 short natural queries, 2 mixed-intent queries, 2 abbreviations, and 2 misspellings, with 15 gold `RAG` and 15 gold `chat`. These categories are not stored in the corpus
- A substantive regulation request has gold `RAG` even when mixed with small talk. Exclude cases requiring history and out-of-scope cases without an agreed binary label
- The Jev request uses exactly one `Choice` question with ID `route` and criteria `RAG`/`chat`. Reject `Noul`, `Score`, other question types, non-`ChoiceAnswer` output, and unexpected answer sections rather than coercing them
- Measure decision correctness and exact-repeat mean pairwise consistency for both routers separately from errors, raw-format validity, usage, latency, and cost. The initial budget is three repeats per query per arm (180 scheduled calls). Do not fold paraphrase or choice-order probes into exact-repeat consistency

The implementation and verification details are in `docs/superpowers/plans/2026-09-23-typesafe-router-experiment.md`. The prior research contract remains in `.knowledge/handoffs/research_state/2026-09-22-typesafe-jev-router.md`

## 2026-09-27 - Freeze the first live baseline and separate Jev tuning

- Chose to freeze `router_exp_001` as the first exploratory baseline (commit `8e0a003`, branch `exp/typesafe-ai`, 2026-09-27) because it compares the production Groq router with pinned `jev-1.13.0` on the same frozen inputs, three repeats, alternating call order, recorded usage, and a dated pricing snapshot
- Chose not to rerun the same 30 queries to strengthen the claim. Further calls should answer new questions through a separately versioned corpus and run instead of turning repeated exposure to the development set into apparent robustness
- Chose to keep production Groq unchanged for now. The baseline establishes that Jev is a strong replacement candidate, but migration needs evidence from unseen boundary cases and paraphrase families rather than another pass over the current corpus
- Chose to optimize Jev criteria in a separate phase because TypeSafe exposes domain policy through `instructions` and `criteria`. The current Jev request preserves the Groq label definitions but omits Groq's two chat examples, so this run compares two concrete router configurations rather than isolating model capability under byte-identical prompts
- Define a boundary case from user intent, not model confidence: it contains vocabulary or context associated with the opposite route while its independently reviewed intent remains decidable. A regulation lookup, condition, procedure, limit, right, or obligation is `RAG`; social conversation, personal reaction, or a request for opinion remains `chat` even when it mentions academic terms; a substantive regulation request wins in mixed intent
- Treat paraphrases as meaning-preserving variants with the same facts, intent, and gold label. A wording change that turns a statement into a request for a regulation is a contrast pair, not a paraphrase. Keep every family in one data split and report both label invariance and all-members-correct so a consistently wrong family cannot look robust

Assumptions:

- Router latency around the observed Jev median is acceptable for the product, while tail latency, correctness, and route-error direction remain visible as separate acceptance dimensions
- `RAG -> chat` is the higher-risk error because it skips retrieval; the first run observed none in either arm, but the corpus is too small to estimate a production rate
- Pricing conclusions apply to the recorded router calls and published rates only; they do not include downstream retrieval, answer generation, retries, fallbacks, or account-specific billing differences
- The binary `RAG`/`chat` policy remains the production contract. History-dependent and unresolved out-of-scope inputs need an explicit policy before joining the scored corpus

Failed approaches:

- Tried: treating exact-repeat consistency as evidence of robustness -> Failed because: both routers repeated their own wrong decision three times -> Avoid when: claiming stability across paraphrases, time, load, or unseen inputs
- Tried: using a low Jev confidence threshold with Groq fallback as an immediate accuracy improvement -> Failed because: the same threshold catches both the Jev error on id 18 and the Jev-correct/Groq-wrong id 15, leaving aggregate correctness unchanged while adding calls and cost -> Avoid when: selecting a fallback threshold from this 30-query run alone
- Tried: describing the run as a byte-identical prompt comparison -> Failed because: Jev receives typed instructions and criteria while Groq receives a text prompt containing two chat examples -> Avoid when: attributing the observed result purely to model architecture

Nuances agreed with the user:

- The baseline is closed as evidence for this exact configuration, not as a production-readiness certificate
- Jev does not need to beat Groq's median latency if its response time meets the product budget and its lower tail variance, cost, and correctness satisfy the agreed acceptance gates
- Future evaluation should compare production Groq, the frozen Jev criteria, and Jev criteria tuned only on development data against the same unseen test set
- The corpus should expand with new ordinary cases, independently authored boundary cases, paraphrase families, abbreviations, missing diacritics, typos, negation, and mixed intent. Variants derived from ids 15, 17, and 18 belong in development data, not the held-out claim
