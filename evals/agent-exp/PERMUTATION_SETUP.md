# Permutation experiment

This experiment replays frozen gate cases with controlled input schedules

- `repeat` replays one exact input several times
- `candidate-order` rotates the presented candidate list
- `seed-order` rotates hop-0 seed parents and rebuilds the compact observation
- `combined` independently combines seed and candidate rotations

The runner does not retrieve articles, run the full loop, synthesize answers, or use RAGAS. Labels never enter policy input, and schedules never read acceptable article IDs

## Prerequisites and commands

Run the initial-selection workflow first so a snapshot and approved cases exist. Permutation replay uses only those files and does not initialize a model for the `first` policy

Run from the repository root

```bash
python evals/agent-exp/scripts/run_experiments.py run \
  --experiment permutation \
  --condition repeat \
  --seeds "$SNAPSHOT_PATH" \
  --cases "$CASES_PATH" \
  --policies first llm \
  --repeats 3 \
  --output-root "$RESULTS_ROOT"

python evals/agent-exp/scripts/run_experiments.py run \
  --experiment permutation \
  --condition candidate-order \
  --schedule rotate \
  --seeds "$SNAPSHOT_PATH" \
  --cases "$CASES_PATH" \
  --policies first llm \
  --repeats 3 \
  --output-root "$RESULTS_ROOT"

python evals/agent-exp/scripts/run_experiments.py run \
  --experiment permutation \
  --condition seed-order \
  --schedule rotate \
  --seeds "$SNAPSHOT_PATH" \
  --cases "$CASES_PATH" \
  --policies first llm \
  --repeats 3 \
  --output-root "$RESULTS_ROOT"
```

`--condition` is required for permutation. `repeat` uses the original schedule; the ordering conditions require `--schedule rotate`. `--repeats` defaults to 3 for permutation and 1 for initial-selection. It records a configuration choice and is not a statistical guarantee. The optional `combined` condition uses the Cartesian product of independent seed and candidate rotations

Every scheduled input is reconstructed and validated before the first model client is initialized. The manifest is saved before policy execution, and each policy receives one call per saved trial

## Schedule and later-hop rules

For `n` candidates, `s` full seed contexts, and `R` repeats, the expected counts are:

| Condition | Seed order | Candidate order | Trials |
| --- | --- | --- | --- |
| `repeat` | original | original | `R` |
| `candidate-order` | original | `n` cyclic rotations | `nR` |
| `seed-order` | `s` cyclic rotations | original | `sR` |
| `combined` | `s` cyclic rotations | `n` cyclic rotations | `nsR` |

Candidate-order keeps observation bytes and changes only the ordered candidate list. Seed-order and combined reorder whole frozen seed contexts, rebuild the observation through the existing normalization/frontier helpers, verify candidate membership is unchanged, and preserve the original candidate list for seed-order

Duplicate or unparseable seed texts and membership-only IDs remain represented during reconstruction. A full policy input identity includes the case ID, observation hash, and ordered candidate tuple. Context-index orders remain distinct even when they render identical observation bytes; `identical_inputs` groups only orders with all three identity components equal, so combined trials with different candidate orders stay separate

Repeat and candidate-order can replay a saved later-hop case. It must have a unique case ID, matching dataset/question/snapshot references, a positive `source_hop`, and non-empty `source_run_id` and `source_policy`. Seed-order and combined require a hop-0 case with full seed contexts. Later-hop cases are recorded with `later_hop_seed_order` exclusion for those conditions. If no runnable cases remain, the runner exits before creating a run or initializing a client

## Metrics

Summaries are recomputed from `manifest.json` and compact `results_<policy>.jsonl` files. Debug files are not required

- Selection and Decision Accuracy are reported per case and condition, then case-macro averaged. STOP cases have undefined Selection Accuracy, represented by `null`
- Question macro averages defined case scores within each dataset/question identity, then averages questions. A source hop is a view of cases, not a new question, and pair counts are never pooled to create a question mean. Source-hop strata and original-question counts are retained
- Single-gold follow cases are grouped by candidate count and 1-based gold position. Errors and missing trials remain in the scheduled denominator. Multi-gold cases are reported separately without choosing a preferred position
- Comparable position cohorts contain only case IDs scheduled at every compared position. Output failures do not decide cohort membership
- Selected-position distributions count valid selections by position and STOP, with scheduled-to-valid coverage beside the valid-output denominator. Errors and missing rows are never interpreted as STOP. First-position Selection Rate includes valid STOP outputs for multi-candidate cases
- Permutation consistency compares distinct permutations within case and repeat ID by STOP or selected article ID. Repeat consistency compares repeated calls for one exact input. Both retain scheduled and valid pair coverage, group and case eligibility, pooled ratios, and hierarchical case means. Case means average defined group values, then question means average defined case values
- Candidate-count exclusions and groups with fewer than two valid outputs remain excluded from consistency means. Their ratios are `null` with explicit eligible/excluded counts, never an invented zero
- Policy agreement compares shared trial IDs only when both policy outputs are valid, with coverage and correctness wins/losses/ties

Consistency does not establish correctness. A stable wrong article and an always-stop policy can be highly consistent while being wrong. Accuracy, gold-position strata, and selected-position distributions must be read together

## Artifacts and limits

Permutation manifests use schema version 2 and save the effective configuration, exact trials, expected trial and policy-result counts, source hashes, code hashes, and specification provenance. Results stay in separate policy files and are appended after each trial. Interrupted runs retain completed rows and are marked incomplete; errors and missing rows remain in scheduled denominators

The gate derives its grammar from the presented candidate list, so candidate-order results cover the current list-and-grammar path and do not isolate prompt bias from grammar effects. Repeated or rotated trials are not new questions, and a small frozen case set cannot support generalization claims. Live ordering evaluation requires a repeat-control run with matching inputs and settings; this runner never adds rehearsal calls
