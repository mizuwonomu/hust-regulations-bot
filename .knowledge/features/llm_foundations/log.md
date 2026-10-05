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
