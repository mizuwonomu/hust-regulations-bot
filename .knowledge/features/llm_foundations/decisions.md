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

