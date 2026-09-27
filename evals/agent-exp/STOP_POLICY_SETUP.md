# STOP policy experiment

This package measures whether the citation gate keeps following candidates when no remaining citation edge is needed

- `first` always follows the first candidate without creating a model client
- `llm` calls the existing gate once per scheduled input

The STOP evaluation does not run the full loop, synthesize answers or use RAGAS. Its source-run exporter fetches only the one article selected by a frozen FOLLOW decision to create a successor state; it does not call the gate again or advance a second edge

## Corpus source and ownership

The current `evals/datasets/corpus_stop_policy.json` is provisional. Before capture, its author must confirm the source, ownership or permission to use it, and that it is intended for this evaluation. Replace the path in the commands below if a different approved source is selected

Source hashes establish which bytes were used; they do not establish ownership, label quality or semantic independence. The evaluator never treats corpus annotations as approved labels

## Phases and where they run

| Command | Environment | Notes |
| --- | --- | --- |
| `capture-corpus` | live retrieval | One retrieval call per corpus row, then one published bundle |
| `inspect-capture` | offline | Reads the bundle only, no corpus or model needed |
| `prepare-hop0` | offline | Freezes nonempty hop-0 requests for one source policy and one repeat |
| `run` on a `later_hop_capture` run | live gate for `llm` | Saves hop-0 decisions for successor-state export; not an evaluation run |
| `export-later-hop` | local article store | Follows each valid FOLLOW result by one edge and writes states or terminal outcomes |
| `import-later-hop` | offline | Verifies source-run provenance and imports exported successor states as drafts |
| `validate-cases` | offline | Labels, groups, splits and readiness only |
| `prepare` | offline | Renders and freezes requests, creates no client |
| `run` | live gate for `llm`, deterministic for `first` | Appends one result per invocation |
| `summarize` | offline | Recomputes the summary and report from saved artifacts |

Offline commands never load an env file. `capture-corpus` accepts `--env-file`, defaults to `.env`, and loads it without overriding variables already exported in the shell

## Corpus input contract

Read as retrieval input, and nothing else:

- `id`
- `user_input`

Never read for retrieval, frontier construction, labels, split assignment, grouping or policy input:

- `response`
- `retrieved_contexts`
- `link`
- `group`
- `type`

Those fields still enter the provenance hashes: each case stores its own raw `source_row_hash` and the whole-file `source_dataset_hash`. Changing an annotation therefore changes provenance without changing retrieval calls, observation, candidates or draft review fields

The internal-article whitelist is the second input and must list the internal article IDs of the corpus

## Label review

`capture-corpus` writes `cases_review.jsonl` as an editable copy of `cases_draft.jsonl` and a `label_review.md` that shows, for every case, the exact question, the exact compact observation, the ordered candidates and the review fields to fill

Approve a label only against the actual gate input: the stored question, compact observation and ordered candidates. Do not use full seed articles or corpus answer and gold annotations to decide whether another citation edge is needed. Draft defaults are `expected_action=unresolved`, `label_status=draft`, `split=unassigned`, `semantic_group_id=null`, `fewshot_overlap=null`, `label_observation_hash=null`

Every review field and its rule:

| Field | Rule |
| --- | --- |
| `expected_action` | `follow` when a remaining citation edge is still needed, `stop` when none is needed, `unresolved` when the observation is not clear enough |
| `acceptable_dieu` | Non-empty subset of `candidates` for `follow`, empty for `stop` and `unresolved` |
| `label_reason` | Non-blank rationale comparing the question with the excerpts |
| `label_status` | `approved` for a finished review, `draft` otherwise |
| `label_observation_hash` | Exact copy of the case `observation_hash`; approving against a different observation fails validation |
| `semantic_group_id` | One group per original question, paraphrase or hop chain |
| `split` | `dev` or `heldout`, identical for every case in the group |
| `fewshot_overlap` | `false` only when the case shares no semantic content with the current few-shot demonstrations |
| `overlap_notes` | What overlaps, when `fewshot_overlap` is `true` |

A STOP label means no current citation edge is needed for the question. It never asserts that the collected articles prove a complete answer

Empty frontiers are `no_candidates`: mechanically excluded, never a positive STOP example, even if a reviewer writes `stop`

### Exclusion reasons

Replay drops a case for exactly one recorded reason, in this order:

1. `no_candidates` - empty frontier
2. `draft` - review not finished
3. `unresolved` - approved as unclear
4. `overlap_unreviewed` - `fewshot_overlap` left `null`
5. `fewshot_overlap` - approved with `fewshot_overlap=true`
6. `split_not_selected` - approved but assigned to the other split

### Grouping and leakage

Related questions, paraphrases, hops and citation paths must share one group and one split. Validation enforces mechanically that a group never crosses `dev` and `heldout`, and that every case of one question keeps the same group and split. Reusing an article ID does not create semantic overlap by itself, and changing IDs does not make a copied case independent

Before `prepare`, a reviewer must inspect semantic groups, paraphrases, citation paths, labels and overlap with the current few-shot examples. Validation checks recorded consistency, not whether the semantic review is correct. Freeze the dev selection before reviewing or running heldout

## Later-hop states

A later-hop state is produced only from a completed `later_hop_capture` source run. `prepare-hop0` selects hop-0 cases with a nonempty frontier from the chosen split, including cases whose action labels are still drafts. It creates one policy and one repeat. Freeze the reviewed case file before preparing the source run; changing it later invalidates the run's source hash. `run` records the decision but does not fetch an article or update collected state

`export-later-hop` verifies that run and, for each valid FOLLOW, fetches the selected article through the local article map/doc store, adds it to the seed's full collected state and rebuilds the next observation/frontier with the agent's production semantics. It writes `later_hop_states.jsonl` for nonempty frontiers and `terminal_outcomes.jsonl` when the successor frontier is empty; terminal outcomes are not STOP cases because no next gate call occurred. STOP, error and missing source results do not produce successor states

Each state binds dataset, question, hop, exact question/observation/candidate order, source run/policy/trial, parent case, followed article id and SHA-256, collected article references and state hash. Import verifies the source run, policy, trial and FOLLOW decision against actual run artifacts and the export bundle file hashes. Imported states start as drafts and must be reviewed in the new case file

- The observation is verified through its explicit state hash, not rebuilt from hop-0 seeds
- The parent case must exist in the base file, must immediately precede this hop, must belong to the same question, and must contain the followed article
- Candidate IDs must belong to the snapshot inventory
- Import writes a new case version with the base cases plus draft later-hop cases and refuses an existing output; the base file is never modified

The source run and exporter advance exactly one edge. They do not run a full loop, synthesize hop-2 inputs or use a prior gate decision to call the gate again. Collected article ids and hashes are retained so future multi-hop extensions can refetch and verify earlier articles

## Readiness

`validate-cases` reports, per split, how many eligible cases exist for the four required classes:

- hop-0 FOLLOW
- hop-0 STOP
- post-follow FOLLOW
- post-follow STOP

Review artifacts stay usable while the matrix is incomplete. `prepare` refuses a split that is missing any of the four classes, so a run can never silently evaluate a corpus without one side of the STOP/FOLLOW question

`prepare-hop0` is the source-decision exception: it requires assigned split and nonempty hop-0 candidates but does not require the four-class matrix. Its manifest uses `run_purpose=later_hop_capture`; its decisions and correctness fields are not a formal STOP-policy measurement. Only run official `prepare` after all four classes are ready

## Artifacts

A capture bundle publishes exactly `capture_manifest.json`, `seeds.json`, `cases_draft.jsonl`, `cases_review.jsonl` and `label_review.md` in one atomic no-replace rename. A prepared evaluation run contains `manifest.json`, `input_traces.jsonl` and a prepared `report.md`; execution adds `results_<policy>.jsonl`, `summary.json` and the final `report.md`. A source run keeps its decisions and a source-only report; it does not write an evaluation summary

- Every destination must be new. CLI writers refuse an existing path without inspecting, rewriting, merging or deleting its contents. If an output path appears during capture, publication fails and preserves that destination
- `seeds.json` is canonical single-line JSON, so its hash is a pure function of its content
- Runs reference the canonical snapshot and case files by path and hash. They never copy those inputs, and they create no trajectory files
- Case files and run directories are diffable text; `debug/` directories are optional and are never required for reload
- `manifest.json` records schema version, run purpose, sources, schedule, exclusions, readiness, executed module hashes, git revision and dirty state, and the effective gate configuration once it is known

Reload verifies source hashes, recomputes the schedule and the baseline traces, recomputes request fingerprints and correctness, and rejects duplicate or unknown results. Deleting `debug/` never changes a summary

## Metrics and denominators

Let F be the number of valid follow-labeled slots and S the number of valid stop-labeled slots

| Metric | Formula |
| --- | --- |
| Follow Accuracy | FOLLOW actions in F / F |
| False Stop Rate | STOP actions in F / F |
| Stop Accuracy | STOP actions in S / S |
| Over-hop Rate | FOLLOW actions in S / S |

- Valid-output rates use only `ok` outcomes inside their label class
- Scheduled Action Accuracy counts every labeled slot; a wrong-article FOLLOW is action-correct, while errors and missing slots are incorrect
- Selection Accuracy uses scheduled follow-labeled slots and requires an acceptable article; STOP, errors and missing slots are incorrect
- Decision Accuracy uses every labeled slot and requires the correct action plus, for FOLLOW, an acceptable article
- Error and missing rates are reported separately for each label over all scheduled slots of that label
- Case-macro metrics compute a defined value per case first, then average cases; undefined values are excluded with explicit defined/eligible counts
- Zero denominators produce `null`, never zero or NaN
- Paired `first`/`llm` output separates action agreement from decision-correctness wins, losses and ties on identical trial IDs

Follow Accuracy scores the action only: it says nothing about article selection. Over-hop here means an unnecessary follow decision on a labeled state, not the number of unnecessary fetches in a full trajectory

## Scope and limits

Only the `baseline` variant is executable in this milestone: the current production prompt, observation and thinking settings. Prompt-balanced, zero-shot, observation-context and thinking-on variants are rejected before a run is materialized

Reported figures support or weaken hypotheses about gate behavior on frozen states. They do not establish answer quality, retrieval benefit, heldout generalization, memorization or an internal model mechanism, and they do not replace a full-loop evaluation. Fake-client checks validate request mechanics and configuration recording; they do not establish current server behavior or model judgment

## Commands

Use the absolute interpreter. Each command group starts from the primary checkout and initializes its own paths

### 1. Offline verification

```bash
cd /home/coronny/rag-project
WORKTREE_ROOT="$(git rev-parse --show-toplevel)"

/home/coronny/rag-project/.venv/bin/python -B -m pytest \
  evals/agent-exp/tests/test_stop_policy_contracts.py \
  evals/agent-exp/tests/test_stop_policy_capture.py \
  evals/agent-exp/tests/test_stop_policy_cases.py \
  evals/agent-exp/tests/test_stop_policy_schedule.py \
  evals/agent-exp/tests/test_stop_policy_artifacts.py \
  evals/agent-exp/tests/test_stop_policy_metrics.py \
  evals/agent-exp/tests/test_stop_policy_later_hop.py \
  evals/agent-exp/tests/test_stop_policy_cli.py \
  -q --confcutdir=evals/agent-exp/tests -p no:cacheprovider

/home/coronny/rag-project/.venv/bin/python -B -m pytest evals/agent-exp/tests \
  -q --confcutdir=evals/agent-exp/tests -p no:cacheprovider

/home/coronny/rag-project/.venv/bin/python -B \
  evals/agent-exp/scripts/stop_policy_eval/cli.py --help

/home/coronny/rag-project/.venv/bin/python -B \
  evals/agent-exp/scripts/stop_policy_eval/cli.py prepare-hop0 --help

/home/coronny/rag-project/.venv/bin/python -B \
  evals/agent-exp/scripts/stop_policy_eval/cli.py export-later-hop --help
```

### 2. Live corpus capture for label review

`capture-corpus` calls live retrieval and was not run during this implementation. First confirm the corpus source and permission to use it. The output paths must be new

The capture runtime rewrites each question through the configured `ChatGroq` client before retrieval. The env file supplies its provider credential (normally `GROQ_API_KEY`) when that value is not already exported; seed serialization itself does not require an env file

```bash
cd /home/coronny/rag-project
WORKTREE_ROOT="$(git rev-parse --show-toplevel)"
CORPUS_PATH="/home/coronny/rag-project/evals/datasets/corpus_stop_policy.json"
ENV_FILE="$WORKTREE_ROOT/.env"
test -f "$CORPUS_PATH"
test -f "$ENV_FILE"
CAPTURE_TAG="$(date -u +%Y%m%dT%H%M%S%N)"
CAPTURE_DIR="$WORKTREE_ROOT/evals/agent-exp/snapshots/stop_policy/${CAPTURE_TAG}_review"
REVIEW_CASES="$CAPTURE_DIR/cases_review.jsonl"
test ! -e "$CAPTURE_DIR"

/home/coronny/rag-project/.venv/bin/python -B \
  "$WORKTREE_ROOT/evals/agent-exp/scripts/stop_policy_eval/cli.py" capture-corpus \
  --env-file "$ENV_FILE" \
  --dataset "$CORPUS_PATH" \
  --internal-dieu "$WORKTREE_ROOT/evals/agent-exp/datasets/internal_dieu_fixed.json" \
  --output-dir "$CAPTURE_DIR"

/home/coronny/rag-project/.venv/bin/python -B \
  "$WORKTREE_ROOT/evals/agent-exp/scripts/stop_policy_eval/cli.py" inspect-capture \
  --capture-dir "$CAPTURE_DIR"

printf 'Review this file: %s\n' "$REVIEW_CASES"
```

The CLI prints the paths to `seeds.json`, `cases_draft.jsonl`, `cases_review.jsonl` and `label_review.md`. Review `label_review.md` and edit the file named by `REVIEW_CASES`. Assign the approved split before preparing the source run. Action labels may remain draft for that source run

### 3. Prepare and run one hop-0 source decision

This source run selects only hop-0 rows in the chosen split with nonempty candidates. It uses one policy and one repeat. `prepare-hop0` is offline; `run` with `llm` calls the live gate and was not run during this implementation. The run purpose is `later_hop_capture`, so it writes decisions without an evaluation summary

```bash
cd /home/coronny/rag-project
WORKTREE_ROOT="$(git rev-parse --show-toplevel)"
CORPUS_PATH="/home/coronny/rag-project/evals/datasets/corpus_stop_policy.json"
test -f "$CORPUS_PATH"
read -r -p "Paste the capture directory printed by capture-corpus: " CAPTURE_DIR
REVIEW_CASES="$CAPTURE_DIR/cases_review.jsonl"
test -f "$REVIEW_CASES"
test -f "$CAPTURE_DIR/seeds.json"

SOURCE_RUN_TAG="$(date -u +%Y%m%dT%H%M%S%N)"
SOURCE_RUN_DIR="$WORKTREE_ROOT/evals/agent-exp/results/stop-policy/${SOURCE_RUN_TAG}_hop0_source_dev"
test ! -e "$SOURCE_RUN_DIR"

/home/coronny/rag-project/.venv/bin/python -B \
  "$WORKTREE_ROOT/evals/agent-exp/scripts/stop_policy_eval/cli.py" prepare-hop0 \
  --seeds "$CAPTURE_DIR/seeds.json" \
  --corpus "$CORPUS_PATH" \
  --cases "$REVIEW_CASES" \
  --split dev \
  --policy llm \
  --output-dir "$SOURCE_RUN_DIR"

/home/coronny/rag-project/.venv/bin/python -B \
  "$WORKTREE_ROOT/evals/agent-exp/scripts/stop_policy_eval/cli.py" run \
  --run-dir "$SOURCE_RUN_DIR"
```

### 4. Export one-edge successor states and import them

`export-later-hop` reads the completed source run, fetches only the article selected by each valid FOLLOW from the local article map/doc store, then rebuilds the next observation and ordered frontier. It does not call the gate again. Empty successor frontiers go into `terminal_outcomes.jsonl` and never become STOP cases. The exporter was not run against the local article store during this implementation; tests inject a fake article fetcher

```bash
cd /home/coronny/rag-project
WORKTREE_ROOT="$(git rev-parse --show-toplevel)"
CORPUS_PATH="$WORKTREE_ROOT/evals/datasets/corpus_stop_policy.json"
read -r -p "Paste the capture directory: " CAPTURE_DIR
REVIEW_CASES="$CAPTURE_DIR/cases_review.jsonl"
read -r -p "Paste the hop-0 source run directory: " SOURCE_RUN_DIR
test -f "$CAPTURE_DIR/seeds.json"
test -f "$REVIEW_CASES"
test -f "$SOURCE_RUN_DIR/manifest.json"

EXPORT_TAG="$(date -u +%Y%m%dT%H%M%S%N)"
EXPORT_DIR="$WORKTREE_ROOT/evals/agent-exp/snapshots/stop_policy/${EXPORT_TAG}_hop1_export"
test ! -e "$EXPORT_DIR"

/home/coronny/rag-project/.venv/bin/python -B \
  "$WORKTREE_ROOT/evals/agent-exp/scripts/stop_policy_eval/cli.py" export-later-hop \
  --run-dir "$SOURCE_RUN_DIR" \
  --output-dir "$EXPORT_DIR"

LATER_HOP_CASES="$WORKTREE_ROOT/evals/agent-exp/snapshots/stop_policy/${EXPORT_TAG}_review_v2.jsonl"
test ! -e "$LATER_HOP_CASES"

/home/coronny/rag-project/.venv/bin/python -B \
  "$WORKTREE_ROOT/evals/agent-exp/scripts/stop_policy_eval/cli.py" import-later-hop \
  --seeds "$CAPTURE_DIR/seeds.json" \
  --cases "$REVIEW_CASES" \
  --states "$EXPORT_DIR/later_hop_states.jsonl" \
  --source-run-dir "$SOURCE_RUN_DIR" \
  --output "$LATER_HOP_CASES"

REVIEW_CASES="$LATER_HOP_CASES"
printf 'Review the new rows in: %s\n' "$REVIEW_CASES"
```

Imported rows start as drafts. Review their action, acceptable articles, rationale, semantic group, split and few-shot overlap in the new `*_review_v2.jsonl` file. The importer checks that the source run, policy, trial and followed article match actual source artifacts

### 5. Validate the complete reviewed case file

`validate-cases` is offline. Run it after reviewing the imported rows; it reports readiness but does not run a model

```bash
cd /home/coronny/rag-project
WORKTREE_ROOT="$(git rev-parse --show-toplevel)"
CORPUS_PATH="/home/coronny/rag-project/evals/datasets/corpus_stop_policy.json"
read -r -p "Paste the capture directory: " CAPTURE_DIR
REVIEW_CASES="$CAPTURE_DIR/cases_review.jsonl"
read -r -p "Paste the reviewed *_review_v2.jsonl path: " REVIEW_CASES
test -f "$CAPTURE_DIR/seeds.json"
test -f "$CORPUS_PATH"
test -f "$REVIEW_CASES"

/home/coronny/rag-project/.venv/bin/python -B \
  "$WORKTREE_ROOT/evals/agent-exp/scripts/stop_policy_eval/cli.py" validate-cases \
  --seeds "$CAPTURE_DIR/seeds.json" \
  --corpus "$CORPUS_PATH" \
  --cases "$REVIEW_CASES"
```

Review the readiness matrix. Continue only when hop-0 FOLLOW, hop-0 STOP, post-follow FOLLOW and post-follow STOP are all ready in the selected split. If any class is missing, add/review appropriate cases before official `prepare`

### 6. Official dev evaluation after readiness

`prepare` and `summarize` are offline. `run` with `llm` calls the live gate model and was not run during this implementation; `first` is deterministic. Use the same frozen corpus and reviewed case file that passed validation

```bash
cd /home/coronny/rag-project
WORKTREE_ROOT="$(git rev-parse --show-toplevel)"
CORPUS_PATH="/home/coronny/rag-project/evals/datasets/corpus_stop_policy.json"
read -r -p "Paste the capture directory: " CAPTURE_DIR
read -r -p "Paste the reviewed *_review_v2.jsonl path: " REVIEW_CASES
test -f "$CAPTURE_DIR/seeds.json"
test -f "$CORPUS_PATH"
test -f "$REVIEW_CASES"
RUN_TAG="$(date -u +%Y%m%dT%H%M%S%N)"
RUN_DIR="$WORKTREE_ROOT/evals/agent-exp/results/stop-policy/${RUN_TAG}_baseline_dev"
test ! -e "$RUN_DIR"

/home/coronny/rag-project/.venv/bin/python -B \
  "$WORKTREE_ROOT/evals/agent-exp/scripts/stop_policy_eval/cli.py" prepare \
  --seeds "$CAPTURE_DIR/seeds.json" \
  --corpus "$CORPUS_PATH" \
  --cases "$REVIEW_CASES" \
  --variant baseline \
  --split dev \
  --policies first llm \
  --repeats 3 \
  --output-dir "$RUN_DIR"

/home/coronny/rag-project/.venv/bin/python -B \
  "$WORKTREE_ROOT/evals/agent-exp/scripts/stop_policy_eval/cli.py" run \
  --run-dir "$RUN_DIR"

/home/coronny/rag-project/.venv/bin/python -B \
  "$WORKTREE_ROOT/evals/agent-exp/scripts/stop_policy_eval/cli.py" summarize \
  --run-dir "$RUN_DIR"
```

Do not treat a heldout command as appropriate until dev selection is frozen and an approved heldout split exists
