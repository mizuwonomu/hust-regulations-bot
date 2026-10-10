# LLM Foundations Decisions

## 2026-10-04 - Parameterized multi-head learning boundary

- The user elected to skip implementing manual attention backward after working through chain rule and an autograd comparison; the October 3 manual-backward delta is superseded by the October 4 multi-head delta, not an implementation prerequisite
- The current teaching architecture uses independent CausalSelfAttention instances in ModuleList, all consuming the same supplied rank-2 X, followed by feature concatenation and a learned output projection. Heads are parallel branches, not sequential layers; their parameter objects must not be shared
- Head dimensions remain independently configurable; this teaching architecture does not require d_model to be divisible by num_heads. Tokenizer/embedding integration, batching, optimized fused projections, residual/norm/FFN and training remain outside this checkpoint
- Output-projection coefficients are unrestricted learned linear weights, not percentages or normalized attention probabilities. Backward splits the concatenated gradient into head slices and sums their contributions at shared X
- Initialization review distinguishes unscaled standard-normal Q/K/V parameters from Xavier-uniform W_O. Different schemes are not inherently a forward defect; Q/K scale can affect softmax saturation, and downstream W_O initialization cannot repair already saturated attention weights
- No initialization change was authorized or implemented during this review; choosing a consistent dimension-aware initialization remains a teaching discussion before training

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


### decisions.md  (why — the expensive part)

#### Chosen approach + why

- Chose Xavier uniform with gain 1 for Q/K/V and output projections (HEAD 132f8a9, branch exp/foundation-model-mechanics, 2026-10-05) because a dimension-aware scale is a simple baseline for the next learning components; unscaled standard-normal projections let activation variance grow with input width
- Kept initialization study bounded to variance propagation and softmax scale because architecture fundamentals are the next dependency; reproducing a large-model initialization recipe is unnecessary for this isolated checkpoint

#### Assumptions it rests on

- For q_j = sum_i x_i w_ij, the simplified derivation assumes independent zero-mean input components of common variance, independent zero-mean weights, and independence between input and weights at initialization: Var(q_j) = d_model * Var(x_i) * Var(w_ij)
- Xavier gain 1 uses weight variance 2 / (fan_in + fan_out), balancing forward and backward scale rather than preserving both exactly for arbitrary rectangular projections
- The scaled-dot-product variance argument additionally approximates independence of Q/K components; shared X, correlations, softmax, residual paths and training limit that approximation
- This choice is a baseline for finite inputs at reasonable scale, not a guarantee of unit activation variance, stable deep training or optimal convergence; revisit when integrating normalization, residual depth and actual training

#### Failed approaches

- Tried: reasoning that scaling W_O by 1/4 could undo multiplying both Q and K by 2 -> Failed because: logits become 4S before nonlinear softmax, so output rescaling cannot generally recover the original value-mixture proportions -> Avoid when: trying to compensate for attention-logit scale after softmax; this was a corrected conceptual proposal, not an executed experiment

#### Nuances agreed with the user

- randn describes standard-normal sampling; Xavier describes a dimension-dependent scale and can use either normal or uniform sampling, so these are not mutually exclusive categories
- The attention divisor sqrt(d_k) controls dot-product width, not arbitrary scale inherited from XW; downstream W_O cannot repair already saturated attention probabilities
- BERT original code uses truncated normal with initializer_range 0.02; GPT-2 public projection code uses normal stddev 0.02, while its report separately describes residual-depth scaling. These are architecture-specific recipes, not evidence that 0.02 is optimal here. Sources: https://github.com/google-research/bert/blob/master/modeling.py ; https://github.com/openai/gpt-2/blob/master/src/model.py ; https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- The learner correctly identified the fourfold logit change and two gradient routes for Y = X + F(X); the mentor clarified that the routes contribute g + J_F(X)^T g. The corrected softmax-compensation explanation has not yet been independently restated by the learner


### decisions.md  (why — the expensive part)

#### Chosen approach + why

- Chose standalone LayerNorm followed by pre-norm attention residual composition (HEAD cde2112, branch exp/foundation-model-mechanics, 2026-10-07) because isolating normalization before composition makes statistics, affine parameters and the skip path independently understandable
- Reused the existing LayerNorm and MHA children because this checkpoint teaches module composition, not a replacement attention implementation; FFN remains the next learning component

#### Assumptions it rests on

- One finite unbatched floating sequence with compatible parameter dtype/device and representable intermediate calculations; no claim about mixed precision, arbitrary extreme inputs or deep-training stability
- LayerNorm computes feature-wise population statistics separately for each token, uses epsilon inside the square root, and shares learned gamma/beta across token positions
- Pre-norm controls the input scale of the transformed branch, not the magnitude of the entire residual stream; any depth-growth argument depends on assumptions about update magnitudes and correlations

#### Failed approaches

- Tried: interpreting zero MHA output as proof that LayerNorm output is zero -> Failed because: the heads and output projection follow normalization, and zero W_O can null their final output even for nonzero normalized inputs -> Avoid when: reasoning backward from a zero residual update to its intermediate activations; this was a corrected conceptual inference, not a failed implementation experiment

#### Nuances agreed with the user

- The learner explicitly distinguished the two forward routes: original X goes directly to addition, while LN(X) enters all attention heads before concatenation and W_O
- Residual addition sums direct and transformed input-gradient contributions; it does not guarantee nonvanishing gradients or preservation of original semantic meaning
- Gamma/beta are learned despite deterministic one/zero initialization; mean and variance are recomputed, and normalization does not create a Gaussian distribution
- The next prerequisite check is why two affine layers need an intervening nonlinear activation, before introducing position-wise FFN


### decisions.md  (why — the expensive part)

#### Chosen approach + why

- Chose to finish one pre-norm decoder block before revisiting input construction (HEAD ed93d7c, branch exp/foundation-model-mechanics, 2026-10-08) because the learner wanted to consolidate existing module composition before introducing text-to-vector mechanics
- Chose a standalone position-wise FFN with exact erf GELU between two affine transformations because the learner wanted to understand smooth activation behavior; the final affine remains unrestricted to permit signed residual updates
- Selected Xavier uniform for both FFN weights and zero biases as a simple dimension-aware teaching baseline, not a proven GELU-optimal initialization; current source reflects that choice
- Kept tokenizer, token embeddings and positional input construction outside the block because the block consumes supplied representations, while those components define how representations are constructed

#### Assumptions it rests on

- This is a single unbatched decoder-only block over finite compatible floating tensors with representable intermediate arithmetic; inherited child contracts remain in force
- Position-wise FFN shares parameters across tokens but does not directly mix rows; the complete block can mix visible earlier positions through causal attention
- A proposed character-level tokenizer and additive learned positional embeddings are introductory teaching choices, not requirements for all Transformers or an approved production tokenizer

#### Failed approaches

- Tried: explaining token independence by saying H had already been computed -> Failed because: independence follows from row-local FFN operations, whereas attention can still couple rows of a previously computed tensor -> Avoid when: inferring dependency structure from execution order rather than the actual function
- Tried: interpreting ReLU as eliminating negative weights and all learning from a token -> Failed because: it gates individual preactivations, while other features and token examples can still contribute gradients -> Avoid when: confusing parameters, activations and upstream loss gradients

#### Nuances agreed with the user

- The learner explained that the second residual retains H, the output of the first residual, so a zero FFN update yields H rather than X
- Two affine layers without an intervening nonlinearity collapse into one affine transformation; nonlinearity permits input-dependent transformations but does not guarantee human-interpretable or increasingly abstract features
- GELU(0) is zero while its local derivative is 0.5; the loss gradient is half the upstream gradient there, not an unconditional nonzero gradient
- Token IDs are vocabulary indices rather than semantic magnitudes. Repeated IDs select the same token-embedding row; additive positional vectors can distinguish occurrences before decoder processing
- Causal masking controls visibility and is not equivalent to positional embeddings; the current block can be reviewed independently of the later positional-input design


### decisions.md  (why — the expensive part)

#### Chosen approach + why

- Chose an ordered 238-code-point JSON vocabulary (commit `86d0046`, branch `exp/foundation-model-mechanics`, 2026-10-10) because an explicit small mapping makes token identity and row selection inspectable before learning byte or subword tokenization
- Chose a standalone character tokenizer with in-memory inverse mappings (commit `b9c32f1`, same branch, 2026-10-10) because text handling should stay separate from tensor computation and vocabulary order must remain stable across calls
- Chose independent tokenizer tests using literal mappings, alternate vocabulary orders and Unicode fixtures (commit `700dbf3`, same branch, 2026-10-10) because round trips alone can conceal a consistently wrong ID assignment or unintended normalization
- Chose persistent token parameters plus learned absolute positional parameters in a separate input module (commit `c34a0a4`, same branch, 2026-10-10) because token identity and sequence position should be observable as separate contributions before decoder processing
- Chose fixed independent output and gradient expectations for the input module (commit `1dd6766`, same branch, 2026-10-10) because shape-only tests do not establish correct row selection, positional addition or repeated-token gradient accumulation

#### Assumptions it rests on

- Vocabulary size V and the token-to-ID mapping must match E: the tokenizer's vocabulary determines the number of embedding rows, and every assigned token ID must select the row with that same token meaning
- The same integer ID in two tokenizers may denote different tokens, even if their vocabulary sizes match. The module's computation can be reused, but already trained weights cannot automatically be reused. A different mapping requires an explicit compatibility decision, such as a justified row remapping, or newly trained weights
- Character tokens here are Python Unicode code points, not UTF-8 bytes, grapheme clusters or BPE units. Encode/decode preserve code-point spelling without Unicode normalization
- The input checkpoint accepts a single unbatched ID sequence with valid IDs and bounded length; text conversion happens outside it. Compatible floating parameter dtype/device and representable arithmetic remain prerequisites
- Normal initialization with mean zero and std 0.02 is a small-scale teaching baseline. It is not evidence of optimal initialization, trained embeddings or stable language-model training

#### Failed approaches

- Tried: let raw token lookup handle ID validation -> Failed because: negative IDs can select E from the end rather than being rejected -> Avoid when: categorical IDs must stay within the tokenizer's vocabulary; this crosses the tokenizer-to-embedding contract
- Tried: slice position rows and rely on the final addition to reject excessive length -> Failed because: with T_max=1, broadcasting can apply P[0] to every token in an overlong sequence -> Avoid when: tensor broadcasting could hide a violated sequence-length contract
- Tried: assign P[2] to the second token of a two-token sequence -> Failed because: positions are zero-based, so the second token uses P[1] and the slice endpoint T is excluded -> Avoid when: translating one-based language into tensor indices; this was a corrected learner explanation, not a remaining core defect

#### Nuances agreed with the user

- The input module specifically implements token embedding plus learned absolute positional embedding. Reusing its interface does not make it a general implementation of all positional schemes
- Repeated tokens share an E row while their positions select different P rows. Parameters survive forward calls; backward computes gradients, and a separate optimizer would update their values
- Actual core API names and registered attributes take precedence over proposed contract spellings, while expected mathematical behavior remains independent of the implementation
- The learner's first core attempt remains the teaching baseline. A single explicitly requested Luna-high agent patched rank, length and ID-range guards before lookup; this authorization did not expand to constructor/dtype/device policies or automatic production integration
- The next learning roadmap has three checkpoints, chosen to connect the toy character pipeline to model-specific prompt length: first distinguish character/code-point tokenization, UTF-8 byte tokenization and BPE's learned pair merges; then inspect a code/log sample with the chosen model's actual tokenizer, viewing token IDs and counts while distinguishing raw text from chat-template and special-token additions; finally explain how token count affects usable context-window capacity, KV-cache memory and computation
- These roadmap checkpoints are proposed next learning work, not completed implementations or measured real-agent results. KV-cache mechanics and token-count effects should be studied before attributing a particular agent's behavior, cost or latency to them
