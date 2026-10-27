# LLM Foundations Log

## 2026-10-04 - Single-head backward understanding and multi-head source review

The learner implemented parameterized single-head attention and reported passing tests. The mentor read self_attention.py and reviewed a standalone backprop.py exercise with identity projections and X=[[1,0],[0,1],[1,1]]. Independent scalar/matrix arithmetic, without autograd, matched the learner's supplied loss 2.6878788 and W_Q/W_K/W_V gradients. This was a bounded numerical cross-check, not a test-suite execution or a general backward implementation verification

The learner then chose to skip writing manual backward and implement multi-head forward. The mentor read multi_head_attention.py and self_attention.py: separate registered heads receive the same X, outputs concatenate along features, attention matrices stack by head, and W_O maps the concatenation to d_model. No multi-head tests were read or run in this review. The latest discussion concerns initialization: Q/K/V use torch.randn, while W_O uses Xavier uniform. The learner has been asked how query variance changes when d_model grows from 64 to 256 under the simplified independent unit-variance input/weight assumptions; the answer is pending

Current contracts and learning gaps are in handoffs/research_state/2026-10-04-multi-head-attention-learning-delta.md. No core was modified during the source review, and no claim of training stability or live-agent behavior was established

## 2026-09-30 - Stable softmax and forward cross-entropy

The learning path started with tensor construction, shape reading, reduction axes, `keepdim`, and right-aligned broadcasting. The learner used those mechanics to implement a row-wise stable softmax for `[B, V]` logits, subtracting one maximum per row before exponentiation and normalizing by a row denominator. Review focused on why `[B, 1]` carries one scalar per batch row and why rectangular batches are necessary to expose accidental alignment with the vocabulary axis

The next mini-component computed cross-entropy directly from logits. The learner used `gather` to select one target logit per row, applied the stable log-sum-exp identity, and implemented `none`, `sum`, and `mean` reductions. The first attempt exposed a real broadcasting defect: `[B, 1] - [B]` expanded to `[B, B]`. The learner identified the shape interaction and corrected the per-sample path to return `[B]` before reduction

Tests remain isolated under `llm_foundations/tests/`. Softmax checks fixed probabilities, rectangular batches, row normalization and independence, row shifts, large positive and negative logits, single-item vocabularies, dtype and input preservation. Cross-entropy checks fixed losses, underflow resistance, target selection, rectangular shape, row independence, row shifts, all reduction shapes, duplicated-batch behavior, device preservation, and input immutability. Built-in PyTorch operations appear only in tests as independent oracles

Result: both forward mini-components met their checkpoint contracts on revision `1ad3d28`. Fresh isolated verification on 2026-09-30 produced 14 passing softmax tests and 9 passing cross-entropy tests. The learner also explained reduction axes, broadcasting, target indexing, per-sample losses, scalar reductions, and the purpose of max-shift numerical stabilization

### features/llm_foundations/log.md  (feature narrative, durable)

The October 1 research-state delta moved the learning exercise from embedding lookup to causal attention with supplied Q/K/V. It recorded a reviewed embedding lookup shape and learner-reported embedding test completion, rather than a new mentor-run verification of those tests. The session then used separate English contracts and test plans for the mask and attention, leaving the first core implementations to the learner

The learner worked through index generation, singleton dimensions and broadcasting, then correctly predicted the query-row-one mask as `[False, False, True, True]`. Core review separated sequence length from the key feature dimension, distinguished scaling from normalization and connected the output shape to value mixing. The revised main-checkout implementation uses a mask sized by token count on the input device and returns the value mixture together with attention weights

Pi and OMP test suites were compared against the contract and run on CUDA. Tests were then strengthened with independent mask expectations, a fixture with multiple active Q/K features and hand-calculated scores, and separate perturbations of the three nonleading key features. The strengthened tests are now present in the main checkout alongside the learner's own cores, rather than relying on either worktree's replacement implementation

Session verification on 2026-10-02 ran `.venv/bin/python` with pytest, `--confcutdir=llm_foundations/tests`, `-p no:cacheprovider`, the two explicit mask/attention test paths, `-q` and `--tb=short`. With CUDA selected as `cuda:0`, the result was 38 passed: 10 mask cases and 28 softmax-dependency/attention cases. This evidence covers the narrow forward contracts, including the masked-softmax input domain, shape and device preservation, causality and multi-feature score checks. The source is now anchored by mask commit `9d496de`, attention commit `ad40ca0` and test commit `270d79c`

At extraction, the learner confirmed that the remaining checks discussed in the session had already been verified and that the learning checkpoint was complete. This closes the weighted-sum understanding check, CPU fallback verification and rejection of the first-feature-only mutation after adding coverage. These confirmations are learner-reported evidence: no additional commands, outputs or independently repeated results were supplied to the mentor. They complement the mentor's observed CUDA run without converting that run into CPU or mutation evidence


### features/llm_foundations/log.md  (feature narrative, durable)

The initialization discussion connected projection width to query variance, then separated dot-product scaling from softmax normalization and downstream output projection. The learner updated Q/K/V initialization and reported passing tests. At extraction, direct source inspection confirmed Xavier-uniform Q/K/V; the previously reviewed output projection already used Xavier uniform. The repository anchor was HEAD 132f8a9 on exp/foundation-model-mechanics. No tests were rerun by the mentor, and no fresh test count, device result or training-stability claim was established

Single-head and multi-head implementations and their tests were grouped into separate proposed commits at the learner's request. Current Git status shows no pending attention core or test changes. The learner retained implementation ownership throughout; the mentor supplied explanations, source review and commit-message drafts. The next proposed learning sequence is residual connections, LayerNorm and position-wise FFN, followed by a pre-norm decoder block, positional information and a small causal language-model training exercise


### features/llm_foundations/log.md  (feature narrative, durable)

The learner implemented standalone LayerNorm and then the pre-norm attention residual core. Mentor source review found the valid-domain normalization arithmetic and composition consistent with the intended mathematics: persistent registered children, epsilon forwarding, the original input skip, projected attention output addition and preserved autograd paths. The learner demonstrated the distinction between normalized attention input and the untouched skip input through the forward-flow discussion

The learner explicitly confirmed that tests passed for the checkpoint and requested no rerun. This is learner-reported verification; the mentor performed source review, not a fresh test execution or full coverage audit. No new test count, device result or training outcome is asserted. At extraction, HEAD cde2112 on exp/foundation-model-mechanics had a clean working tree before knowledge edits, and the LayerNorm comment now correctly described learned affine coefficients

Implementation status is recorded as complete for standalone LayerNorm and the attention residual sublayer. Learning proceeds to position-wise MLP/FFN; this result does not represent a complete decoder block


### features/llm_foundations/log.md  (feature narrative, durable)

The learner implemented the affine-GELU-affine FFN and reported all its tests passing. Mentor source review found the intended mathematics and persistent learned parameters; documentation support clarified parameter sharing and signed outputs. The later decoder block reused the existing attention residual, a distinct FFN LayerNorm, and the FFN. Source review confirmed correct two-residual composition, epsilon forwarding and unchanged attention probabilities. The learner subsequently applied the proposed decoder docstrings; current source was reread at extraction

At HEAD ed93d7c on exp/foundation-model-mechanics, the working tree was clean before this knowledge update. The decoder core is implemented and source-reviewed. No tests were executed during these reviews: FFN success is learner-reported, and decoder-specific test success was not explicitly supplied in this dialogue. This records an implementation baseline rather than a trained language model

The current learning discussion returns to the origin of decoder input X. An illustrative character-to-ID mapping and embedding lookup were explained, followed by adding a position vector to each token vector. The next session should establish the learner's understanding of vocabulary/encode/decode and learned embedding rows before assigning input-pipeline implementation; it should not resume FFN as the next unfinished core


### features/llm_foundations/log.md  (feature narrative, durable)

The learner supplied the vocabulary, character tokenizer and input-embedding core as independent teaching components. Five local commits preserved the requested asset/core/test order: vocabulary `86d0046`, tokenizer `b9c32f1`, tokenizer tests `700dbf3`, input module `c34a0a4`, and input tests `1dd6766`, on `exp/foundation-model-mechanics`. The vocabulary initializer's filename was corrected to `__init__.py` at the user's request. None of these commits was pushed during this checkpoint

The learner demonstrated token-row sharing for repeated IDs and worked through selecting the first T position rows. The initial one-based answer for the second token was corrected in the dialogue; an independent restatement of that correction was not recorded. Core review found that negative IDs and excessive sequence length could escape rejection, so the explicitly delegated patch added rank, length and range checks before lookup without changing the registered tables or the intended forward calculation

The delegated agent reported a temporary regression probe that failed before the guard patch and passed afterward, including rejection before lookup and valid value/gradient checks. That probe was not added to the repository. Fresh pre-commit verification in the main checkout produced `150 passed in 0.17s` for character-tokenizer tests and `18 passed in 6.20s` for input-embedding tests. The embedding suite selected CPU because CUDA was unavailable. Each suite ran through `.venv/bin/python -m pytest --confcutdir=llm_foundations/tests -p no:cacheprovider`, targeting its respective test file with `-q --tb=short`; the embedding run also used `-s` to expose device selection. No tests were rerun after the initializer rename, following the user's instruction

The committed tokenizer checks establish code-point mapping, inverse conversion, vocabulary-order differences, rejection behavior and state preservation over the exercised fixtures. The committed input checks establish fixed forward values, boundary/singleton behavior, sharing, persistence, serialization and analytical/numerical gradients over the exercised valid inputs. The outcome is an isolated input-construction checkpoint with the requested guard patch; it does not establish a full decoder pipeline, trained language model or causal explanation of real-agent behavior. Outstanding coverage and the next focus are maintained in `tracker.md`


### features/llm_foundations/log.md  (feature narrative, durable)

The learner moved from character/byte/BPE experiments to vocabulary prediction and one-step decoding. The teaching discussion separated tokenizer training from language-model training, then explained how each decoder output row scores the following token. The learner demonstrated the shifted-target interpretation and calculated a small rectangular head projection before implementing the head. Independent head weights were selected to keep the first projection checkpoint separate from weight tying

The learner then implemented greedy selection and categorical sampling with temperature and optional top-k or top-p. Source review examined the selector and its tests, including original-ID mapping, retained probability distributions, generator replay and input preservation. The learner explained why lower-probability tokens remain selectable without filtering, why top-k=1 agrees with a unique greedy maximum, and why top-p retains a variable number of candidates. The review did not execute the suites, following the learner's request while tests were running elsewhere

At extraction, Git confirmed the requested core-before-test sequence: LM head core `d835c6a`, head tests `4654e04`, decoding core `a563852`, and decoding tests `4f79e22`, following tokenizer experiment checkpoint `a41fb7e`. HEAD was `4f79e22` on `exp/foundation-model-mechanics`; the working tree was clean before these knowledge edits. The outcome is committed standalone components and test suites, with source-reviewed educational behavior. No new test count, device result, trained-model quality or end-to-end generation result is asserted by this extraction

The learner explicitly set the next focus to composing a small decoder-only forward model before studying the autoregressive loop and inference. This is a new composition checkpoint rather than another tokenizer implementation or an immediate generation loop. The mentor continues to provide concepts, contracts and review while leaving the first implementation to the learner

### 2026-10-23: Cached generation and benchmark interpretation checkpoint

Subsequent source reviews covered cached single-head, multi-head, decoder block, decoder-only model with position offsets, and generation. Learner-reported passing tests were not rerun by the mentor. The current learning focus is benchmark interpretation rather than the earlier assembly-next status above

Read the recorded notebook and acceptance manifest under `llm_foundations/experiments/kv_cache_benchmark_runs/20261023T112207Z_coordinator_dca30821/`, plus current benchmark source. The notebook records CUDA RTX 3060, two warm-ups and seven repetitions, successful logit comparisons and matching greedy IDs in the displayed cases. Its prompt-sweep controlled-decode speedups are approximately 1.322x, 1.579x, 1.020x for P=8/32/96, B=16; corresponding wrapper speedups are 1.020x, 1.065x, 0.937x. These are historical artifact outputs, not freshly reproduced measurements

The manifest points to sibling `20261023T112225Z_9ed22424/results.json`, absent at inspection, so raw timing samples were not re-audited. The recorded 8.125 MiB pre-model allocation remains unexplained. No claim that growing cache causes the wrapper slowdown or differing quartile widths is established

The user requested a Research State for another agent to plan repeated timing and annotated profiling, plus a cross-session handoff. Planning state: `.knowledge/handoffs/research_state/2026-10-23-kv-cache-profiling-followup.md`. It preserves greedy, separates profiler runs from benchmark latency, and targets selector synchronization, cache concatenation and Transformer suboperations. No benchmark/profiler was run and no core was changed in this interpretation session

### 2026-10-25: Profiling delivered, experiment-isolation migration pending

The previous implementation added semantic source annotations, an opt-in capture harness and an unexecuted teaching notebook. One Pi implementer used provider `openai`, model `gpt-6-luna`, thinking `xhigh`; the coordinator reviewed two correction batches and independently reran the final focused suite on 2026-10-24: `16 passed, 1 warning in 2.33s`, with the matched tiny-model fixture on CUDA. Export/error-policy checks were synthetic; the actual profiler plumbing smoke used one tiny CPU tensor operation. The notebook, real model profiler/CUDA trace, benchmark and sweeps were not executed. These results apply to the delivered source-annotation implementation, not to the proposed isolation migration

The user subsequently questioned direct core diffs and agreed to move instrumentation entirely into `llm_foundations/experiments/`. The selected boundary and granularity limitations are recorded in `decisions.md` under `2026-10-25: Keep profiling instrumentation inside experiments`. The migration has not started. The user requested only a cross-session handoff for the move; no code, tests, notebook or contract was changed during this handoff

Existing contract/test plan: `docs/superpower/2026-10-24-kv-cache-profiling-contract.md` and `docs/superpower/2026-10-24-kv-cache-profiling-test-plan.md`. Implementation evidence: `llm_foundations/experiments/kv_cache_profiling_implementation_report.md`. Resume by reviewing the current annotation diff and revising the contract/test plan for an experiment-owned scoped instrumentation adapter. The old planning-only research state and source-annotation contract are historical context, not the current target architecture

### features/llm_foundations/log.md  (feature narrative, durable)

At HEAD `d54deb2` on `exp/foundation-model-mechanics`, the user checkpointed benchmark work in the requested core-before-tests-before-results order: `67b3b8a` contains the harness and teaching notebook, `5dd1387` contains focused tests, and `d54deb2` contains the existing acceptance manifest and executed notebook. No push occurred. Profiling artifacts and existing knowledge edits were left outside these commits

Fresh pre-commit verification ran `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -s --confcutdir=llm_foundations/experiments -p no:cacheprovider llm_foundations/experiments/test_kv_cache_benchmark.py` and returned `8 passed, 1 warning in 7.32s`. The captured output did not identify the selected device, so no fresh CUDA claim is made. This suite exercises small workloads; it did not rerun the historical seven-repeat notebook benchmark. Both notebook JSON files parsed with zero saved error outputs, and each staged commit passed whitespace checks

The archive was retained as historical evidence after checking its references and hashes. The test-file hash matched the manifest, while the current harness and source-notebook hashes did not. These checks establish which evidence can be attributed to the snapshot, rather than reproducing its timing results. Current debt is tracked separately in tracker.md

During this extraction, the separate profiling implementation report was read only to reconcile the older migration-pending note. It now reports experiment-owned instrumentation and offline report rendering implemented, while profiling files remain untracked. That statement is report-derived: no profiling source audit, test execution, or trace verification was performed here. The earlier pending status is historical, not proof that migration work still needs to be written
