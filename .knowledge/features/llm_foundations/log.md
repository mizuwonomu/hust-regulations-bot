# LLM Foundations Log

## 2026-09-30 - Stable softmax and forward cross-entropy

The learning path started with tensor construction, shape reading, reduction axes, `keepdim`, and right-aligned broadcasting. The learner used those mechanics to implement a row-wise stable softmax for `[B, V]` logits, subtracting one maximum per row before exponentiation and normalizing by a row denominator. Review focused on why `[B, 1]` carries one scalar per batch row and why rectangular batches are necessary to expose accidental alignment with the vocabulary axis

The next mini-component computed cross-entropy directly from logits. The learner used `gather` to select one target logit per row, applied the stable log-sum-exp identity, and implemented `none`, `sum`, and `mean` reductions. The first attempt exposed a real broadcasting defect: `[B, 1] - [B]` expanded to `[B, B]`. The learner identified the shape interaction and corrected the per-sample path to return `[B]` before reduction

Tests remain isolated under `llm_foundations/tests/`. Softmax checks fixed probabilities, rectangular batches, row normalization and independence, row shifts, large positive and negative logits, single-item vocabularies, dtype and input preservation. Cross-entropy checks fixed losses, underflow resistance, target selection, rectangular shape, row independence, row shifts, all reduction shapes, duplicated-batch behavior, device preservation, and input immutability. Built-in PyTorch operations appear only in tests as independent oracles

Result: both forward mini-components met their checkpoint contracts on revision `1ad3d28`. Fresh isolated verification on 2026-09-30 produced 14 passing softmax tests and 9 passing cross-entropy tests. The learner also explained reduction axes, broadcasting, target indexing, per-sample losses, scalar reductions, and the purpose of max-shift numerical stabilization

