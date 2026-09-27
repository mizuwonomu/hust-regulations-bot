# Router Experiment Log

## 2026-09-23

- Research state is recorded in `.knowledge/handoffs/research_state/2026-09-22-typesafe-jev-router.md`; accepted corpus and metric decisions are in `decisions.md`
- An English implementation plan was written at `docs/superpowers/plans/2026-09-23-typesafe-router-experiment.md`. No corpus, harness, tests, or live benchmark has been implemented or run
- Next work, after plan review: implement the strict 30-case corpus loader and smoke tests, then the two routing adapters, budgeted runner, and offline metrics in plan order
- Account access to the exact production Groq model `qwen/qwen3.6-27b` and its API pricing were confirmed by the project owner on 2026-09-25, so a live comparison can use it as the baseline without a successor label

## 2026-09-25 - Implementation review

- The isolated corpus, adapters, runner, metrics, and router tests are present as uncommitted files. Offline verification passed: 117 tests and a 180-call dry-run schedule; no live model call was made
- The implementation is not ready for a live run. `run.py` treats a missing `trials.jsonl` as zero completed trials and can repeat all paid calls; it also truncates any malformed final JSON line without distinguishing a torn write from a complete corrupted record
- `metrics.py` does not emit the plan's complete per-case outcomes, does not bind each trial's gold label back to the hashed corpus, and may price usage against the requested model even when the recorded served model differs
- Secondary gaps: the Jev request guard checks criteria keys but not criteria definitions/instructions, and the production fallback parity check can be fooled by matching source text in comments or inactive branches
- The exact baseline `qwen/qwen3.6-27b` was deprecated for Groq free/developer tiers on 2026-09-14; enterprise committed-spend access is the documented exception. A separate baseline decision is still required if the account cannot serve it

## 2026-09-27 - Live baseline captured

The reviewed recovery, request-contract, parity, and scoring blockers were repaired before the live run. The committed harness now validates the corpus and adapter contracts, alternates arm order, flushes every trial, refuses unsafe resume state, binds scored rows to the hashed corpus, and prices only complete usage against the model actually served. Router tests are isolated from shared dotenv and database fixtures.

`router_exp_001` ran 180 calls: 30 cases, two arms, and three repeats. Both arms recorded 90 valid decisions with no API errors, missing trials, or invalid formats. Each achieved 87/90 decision correctness (96.67 percent), 29/30 case correctness, and 1.0 exact-repeat consistency. Groq misrouted chat id 15 as `RAG` in all repeats; Jev misrouted chat id 18 as `RAG` in all repeats; id 17 was correctly routed to `chat` by both.

Groq latency was 218.0 ms p50, 476.2 ms p95, and 1124.7 ms maximum. Jev latency was 262.1 ms p50, 299.8 ms p95, and 694.2 ms maximum. Groq was faster in 80 of 90 paired calls, while Jev had better upper-tail latency in this run. Recorded usage and the manifest's dated rates estimated 90-call cost at USD 0.0151182 for Groq and USD 0.00163989 for Jev, about 89.15 percent lower for Jev.

The dataset, harness, tests, and result artifacts were committed separately as `dd3f3bd`, `30a6ceb`, `aea744d`, and `8e0a003`. Fresh isolated verification on 2026-09-27 passed all 137 router tests with one upstream beta API warning. The first baseline is now frozen; the next research phase is corpus expansion and Jev-specific criteria evaluation under a new manifest and result directory. Open evidence limits and migration work remain in `tracker.md`.
