# LLM Foundations Decisions

## 2026-09-30 - Stable softmax and forward cross-entropy

### Chosen approach and why

- Chose an isolated `llm_foundations/` teaching package for bottom-up implementations (commits `3221c34`, `aa4dd72`, `c28aeeb`, and `1ad3d28`, branch `exp/foundation-model-mechanics`, 2026-09-30) because learning code must remain separate from the production RAG chain and must not be mistaken for a framework migration or agent feature
- Chose learner-first ownership: the learner writes the first core attempt, while models provide mathematical contracts, test plans, review, and graduated hints because replacing the attempt with agent-written code would defeat the checkpoint
- Chose max-shifted row-wise softmax and direct log-sum-exp cross-entropy because both preserve the intended mathematics while avoiding overflow and the `log(0)` failure caused by probability-first cross-entropy
- Chose black-box behavioral tests against PyTorch oracles plus manual source review because output tests establish the mathematical contract, while source review is still needed to enforce the no-built-in learning boundary

### Assumptions it rests on

- These checkpoints cover finite rank-2 logits only, with nonempty batch and vocabulary dimensions
- Cross-entropy targets are valid `torch.long` class IDs, one per row, and every ID is within the vocabulary range
- The work is an offline learning surface with no production imports, network calls, database fixtures, or live-agent claims
- The dependency order remains softmax and cross-entropy, then embeddings, attention, Transformer decoder, causal language modeling, and decoding

### Failed approaches

- Tried: combine row maxima shaped `[B, 1]` directly with selected logits shaped `[B]` in cross-entropy -> Failed because: right-aligned broadcasting produced `[B, B]` and coupled every sample with other samples' targets -> Avoid when: mixing row-reduction tensors with per-sample vectors without aligning the singleton dimension explicitly

### Nuances agreed with the user

- `[B, V]` combined with `[B, 1]` applies one value across each row; `[B]` aligns with the final dimension and can either fail when `B != V` or silently apply column-wise when `B == V`
- `keepdim=True` preserves a reduction axis for broadcasting; it does not create row independence, which comes from reducing across the vocabulary axis while leaving the batch axis intact
- Targets are indices, not numeric labels to broadcast into logits; advanced indexing or `gather` must select exactly one logit from each row
- Cross-entropy first produces one loss per sample with shape `[B]`; `none` preserves it, while `sum` and `mean` return scalar tensors with shape `[]`
- PyTorch softmax and cross-entropy are allowed only as independent test oracles; the learning cores must express the stable calculations directly
- Softmax tests cover CPU float32 and float64; cross-entropy tests use float64 on CUDA when available and fall back to CPU

### decisions.md  (why - the expensive part)

#### Chosen approach and why

- Chose separate learner-owned causal mask and attention cores (mask `9d496de`, attention `ad40ca0`, tests `270d79c`; branch `exp/foundation-model-mechanics`, 2026-10-02) because isolating position visibility from value mixing lets the learner inspect each prerequisite without introducing a decoder, learned projections or production changes
- Chose a boolean mask with True meaning blocked and a visible diagonal because the exercise must exclude future keys while retaining at least one valid key per query; an all-negative-infinity row would break the stable-softmax calculation
- Chose to reuse the existing stable softmax after masking scores because attention extends a previously learned component instead of hiding normalization inside built-in attention; this extends its exercised input domain to negative infinity with a finite score remaining in every row
- Chose to inherit the mask device from Q and return `(output, weights)` because device consistency and observable weights are part of the attention contract; the core must follow the supplied tensors rather than select hardware independently
- Chose independent scalar index expectations for masks and hand-calculated multi-feature scores for attention because reusing the core's construction or testing only one active feature can conceal an incorrect implementation

#### Assumptions it rests on

- This checkpoint accepts one sequence and one head with supplied Q/K/V, positive dimensions, a common T, matching Q/K feature dimensions, a common dtype/device and finite inputs whose computed scores remain representable
- The exercised numerical dtypes are float32 and float64; no guarantee is made for arbitrary extreme finite values or mixed precision
- Tests choose CUDA when `torch.cuda.is_available()` is true, otherwise CPU; fixtures and expected tensors use the selected device, and a failure on CUDA is reported rather than retried as a successful CPU run
- The scope is forward-only; batching, learned projections, positional encoding, padding/arbitrary masks, dropout, KV caching, backward tests and training are separate checkpoints
- Invalid shapes, empty dimensions, mismatched inputs, non-finite Q/K/V and all-masked rows have no specified validation or exception contract

#### Failed approaches

- Tried: use `arange` with an end of `seq_len - 1` -> Failed because: the exclusive endpoint omitted the final position and produced a mask smaller than the sequence -> Avoid when: translating an inclusive last index into a range endpoint
- Tried: construct the mask using `k.shape[1]` -> Failed because: that dimension is d_k, while query/key scores have T rows and T columns; rectangular fixtures exposed the mismatch -> Avoid when: treating token count and feature count as interchangeable
- Tried: leave the attention mask on its default CPU device or hardcode CUDA -> Failed because: the mask must share the input device, and hardcoded CUDA neither follows CPU inputs nor provides availability fallback -> Avoid when: placing hardware selection inside an operation that consumes existing tensors
- Tried: return only the value mixture -> Failed because: the agreed caller contract also needs attention weights for inspection and verification -> Avoid when: relying on a tuple annotation without matching the actual returned container
- Tried: use equal-score fixtures and unequal-score fixtures with only the first Q/K feature active -> Failed because: an in-memory implementation using only the first feature still passed all 30 tests in each original worktree -> Avoid when: assuming d_k greater than one proves that every feature contributes to the tested dot product

#### Nuances agreed with the user

- Mask rows denote query positions and columns denote key positions; broadcasting a row-index column and a column-index row compares every pair without a Python loop or first splitting an existing score matrix
- Scaling divides logits by sqrt(d_k); normalization into probabilities happens later over the key axis. Negative infinity replaces blocked scores before softmax, rather than replacing them with zero or zeroing probabilities without renormalization
- Attention output is a weighted sum of value vectors; each individual output feature is a dot product between the weight row and the corresponding V column. Mixing removes the key-position dimension and retains d_v features
- The scale variance argument assumes independent components with mean zero and variance one; it is not evidence that real Q/K always satisfy those assumptions
- Keep the learner's reviewed core as the implementation source. Passing tests against a different worktree's corrected core does not verify the learner's main-checkout attempt
- Toy attention behavior and test results do not establish a mechanism for STOP/FOLLOW, few-shot or ordering effects in the real agent
