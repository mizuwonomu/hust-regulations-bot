# Initial selection experiment

This directory measures two decisions on the same frozen hop-0 observation:

- `first` follows the first candidate without creating an LLM client
- `llm` calls the existing citation gate once and may follow or stop

The experiment does not retrieve articles, run the full loop, synthesize answers, or use RAGAS

For repeat, candidate-order, seed-order, and combined replays on these frozen inputs, see [PERMUTATION_SETUP.md](PERMUTATION_SETUP.md)

## Workflow

1. Supply a reproduced single-pass baseline, its original question dataset, and an independent internal-article whitelist
2. Import the seed snapshot
3. Prepare draft cases
4. Review every observation and edit only the label fields
5. Run offline policy checks or the approved experiment
6. Rebuild a summary from the saved run when needed

The baseline must explicitly identify itself as single-pass with `config.agent: false`, contain `results[*].retrieved_contexts` as ordered full strings, and contain `results[*].hop_scores.retrieved_dieu` as a separate ID list

The dataset is a JSON array with unique IDs and `user_input` strings. The whitelist is a JSON array of unique positive integer article IDs. Whitelist IDs are not inferred from gold labels or contexts

## Commands

Run from the repository root

```bash
python evals/agent-exp/scripts/run_experiments.py import-seeds \
  --baseline "$BASELINE_PATH" \
  --dataset "$DATASET_PATH" \
  --internal-dieu "$WHITELIST_PATH" \
  --output "$SNAPSHOT_PATH"

python evals/agent-exp/scripts/run_experiments.py prepare-cases \
  --seeds "$SNAPSHOT_PATH" \
  --output "$CASES_PATH"

python evals/agent-exp/scripts/run_experiments.py run \
  --experiment initial-selection \
  --seeds "$SNAPSHOT_PATH" \
  --cases "$CASES_PATH" \
  --policies first llm \
  --repeats "$REPEAT_COUNT" \
  --output-root "$RESULTS_ROOT"

python evals/agent-exp/scripts/run_experiments.py summarize --run-dir "$RUN_DIR"
```

Use `--policies first` to verify the deterministic arm without a local gate server

## Label review

`prepare-cases` writes one JSON object per line. Non-empty frontiers start as `unresolved` and `draft`. Empty frontiers are `no_candidates` and are mechanically excluded

For an approved semantic case, set:

- `expected_action` to `follow` or `stop`
- `acceptable_dieu` to a non-empty subset of `candidates` for `follow`, or `[]` for `stop`
- `label_reason` to a non-blank human rationale
- `label_status` to `approved`

Do not put gold labels, answer text, or label rationales into the policy input. The runner projects only `question`, `observation`, and ordered `candidates`

## Artifacts

Each run saves a manifest, the planned shared trials, one result JSONL file per executed policy, and a deterministic summary. Results are appended after each trial

Verbose decisions and logs live under `results/**/debug/` and are ignored by Git. Independent initial replays do not create trajectory files

The summary reports primary accuracy over all scheduled labeled trials, valid-output coverage, first-position selection rate, candidate-count groups, paired agreement, correctness wins/losses/ties, errors, and missing trials. Undefined ratios are `null`

`first_position_selection_rate` is a behavioral metric and is not a correctness claim. A difference from `first` is not evidence of a good decision, and this small diagnostic set cannot support a generalization claim

Inputs and labels remain versioned separately from policy results. Changing a label requires a new case file and a new run
