# Fixed diagnostic report: 13836d8cc7c7

- suite: `fixed-diagnostic-a-v1`
- diagnostic kind: `candidate-relative-position`
- case: `corpus_cross_references:hop0:q-ef2d127de37b`
- status: `complete`
- policies: ['first', 'llm']
- source revision: `35bd7b1`
- execution revision: `d863d1f79607e3da042fe1cc2253b51c83bf7f09`
- independent_question_count = 1
- started at: 2026-09-16T15:50:55.227776+00:00
- ended at: 2026-09-17T07:31:33.229261+00:00

## Result-native evidence

| role | path | sha256 |
| --- | --- | --- |
| experiment definition | embedded in manifest | immutable result evidence |
| case | `evals/agent-exp/datasets/gate_cases_fixed_v1.jsonl` | `330675fde571e925416e5e00d962c54a30e213c4a8b8bc276e911fa5ed92ec4c` |
| snapshot | `evals/agent-exp/snapshots/seeds_v1_fixed.json` | `ca012ef4915895387101fc9b430e8dfa2d6054ab9238802ddc5cf7799cc8366b` |
| inventory | `evals/agent-exp/datasets/internal_dieu_fixed.json` | `da509a47b9c7b82c4717c7df894c9d3bb44eba2fcade8e1e91f466dba5406ca2` |

## Schedule

### Trial schedule

| arm | presented candidates | pair positions | relative order | transform | trace |
| --- | --- | --- | --- | --- | --- |
| `a0-original-control` | [41, 3, 40, 8, 43] | [1, 2] | 41-3 | original | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` |
| `a1-pair-23-3-first` | [8, 3, 41, 40, 43] | [2, 3] | 3-41 | original | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` |
| `a2-pair-23-41-first` | [8, 41, 3, 40, 43] | [2, 3] | 41-3 | original | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` |
| `a3-pair-34-3-first` | [8, 40, 3, 41, 43] | [3, 4] | 3-41 | original | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` |
| `a4-pair-34-41-first` | [8, 40, 41, 3, 43] | [3, 4] | 41-3 | original | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` |

| batch | repeat | trial | arm |
| --- | --- | --- | --- |
| batch-0 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-0:r0` | `a0-original-control` |
| batch-0 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-0:r0` | `a1-pair-23-3-first` |
| batch-0 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-0:r0` | `a2-pair-23-41-first` |
| batch-0 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-0:r0` | `a3-pair-34-3-first` |
| batch-0 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-0:r0` | `a4-pair-34-41-first` |
| batch-0 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-0:r1` | `a0-original-control` |
| batch-0 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-0:r1` | `a1-pair-23-3-first` |
| batch-0 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-0:r1` | `a2-pair-23-41-first` |
| batch-0 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-0:r1` | `a3-pair-34-3-first` |
| batch-0 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-0:r1` | `a4-pair-34-41-first` |
| batch-0 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-0:r2` | `a0-original-control` |
| batch-0 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-0:r2` | `a1-pair-23-3-first` |
| batch-0 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-0:r2` | `a2-pair-23-41-first` |
| batch-0 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-0:r2` | `a3-pair-34-3-first` |
| batch-0 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-0:r2` | `a4-pair-34-41-first` |
| batch-1 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-1:r0` | `a4-pair-34-41-first` |
| batch-1 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-1:r0` | `a3-pair-34-3-first` |
| batch-1 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-1:r0` | `a2-pair-23-41-first` |
| batch-1 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-1:r0` | `a1-pair-23-3-first` |
| batch-1 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-1:r0` | `a0-original-control` |
| batch-1 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-1:r1` | `a4-pair-34-41-first` |
| batch-1 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-1:r1` | `a3-pair-34-3-first` |
| batch-1 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-1:r1` | `a2-pair-23-41-first` |
| batch-1 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-1:r1` | `a1-pair-23-3-first` |
| batch-1 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-1:r1` | `a0-original-control` |
| batch-1 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-1:r2` | `a4-pair-34-41-first` |
| batch-1 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-1:r2` | `a3-pair-34-3-first` |
| batch-1 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-1:r2` | `a2-pair-23-41-first` |
| batch-1 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-1:r2` | `a1-pair-23-3-first` |
| batch-1 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-1:r2` | `a0-original-control` |

## Decisions

### Decisions

| trial | policy | arm | batch | repeat | action | selected article | selected position | input trace | selection correct | decision correct | outcome | latency ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-0:r0` | first | `a0-original-control` | batch-0 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-0:r0` | llm | `a0-original-control` | batch-0 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 1650.2 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-0:r0` | first | `a1-pair-23-3-first` | batch-0 | 0 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-0:r0` | llm | `a1-pair-23-3-first` | batch-0 | 0 | FOLLOW:41 | 41 | 3 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 576.8 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-0:r0` | first | `a2-pair-23-41-first` | batch-0 | 0 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-0:r0` | llm | `a2-pair-23-41-first` | batch-0 | 0 | FOLLOW:41 | 41 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 575.3 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-0:r0` | first | `a3-pair-34-3-first` | batch-0 | 0 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-0:r0` | llm | `a3-pair-34-3-first` | batch-0 | 0 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 574.8 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-0:r0` | first | `a4-pair-34-41-first` | batch-0 | 0 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-0:r0` | llm | `a4-pair-34-41-first` | batch-0 | 0 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 583.4 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-0:r1` | first | `a0-original-control` | batch-0 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-0:r1` | llm | `a0-original-control` | batch-0 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 573.5 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-0:r1` | first | `a1-pair-23-3-first` | batch-0 | 1 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-0:r1` | llm | `a1-pair-23-3-first` | batch-0 | 1 | FOLLOW:41 | 41 | 3 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 580.8 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-0:r1` | first | `a2-pair-23-41-first` | batch-0 | 1 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-0:r1` | llm | `a2-pair-23-41-first` | batch-0 | 1 | FOLLOW:41 | 41 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 574.1 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-0:r1` | first | `a3-pair-34-3-first` | batch-0 | 1 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-0:r1` | llm | `a3-pair-34-3-first` | batch-0 | 1 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 584.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-0:r1` | first | `a4-pair-34-41-first` | batch-0 | 1 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-0:r1` | llm | `a4-pair-34-41-first` | batch-0 | 1 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 578.2 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-0:r2` | first | `a0-original-control` | batch-0 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-0:r2` | llm | `a0-original-control` | batch-0 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 580.3 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-0:r2` | first | `a1-pair-23-3-first` | batch-0 | 2 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-0:r2` | llm | `a1-pair-23-3-first` | batch-0 | 2 | FOLLOW:41 | 41 | 3 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 576.4 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-0:r2` | first | `a2-pair-23-41-first` | batch-0 | 2 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-0:r2` | llm | `a2-pair-23-41-first` | batch-0 | 2 | FOLLOW:41 | 41 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 577.8 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-0:r2` | first | `a3-pair-34-3-first` | batch-0 | 2 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-0:r2` | llm | `a3-pair-34-3-first` | batch-0 | 2 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 581.1 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-0:r2` | first | `a4-pair-34-41-first` | batch-0 | 2 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-0:r2` | llm | `a4-pair-34-41-first` | batch-0 | 2 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 575.4 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-1:r0` | first | `a4-pair-34-41-first` | batch-1 | 0 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-1:r0` | llm | `a4-pair-34-41-first` | batch-1 | 0 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 317.5 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-1:r0` | first | `a3-pair-34-3-first` | batch-1 | 0 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-1:r0` | llm | `a3-pair-34-3-first` | batch-1 | 0 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 1365.4 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-1:r0` | first | `a2-pair-23-41-first` | batch-1 | 0 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-1:r0` | llm | `a2-pair-23-41-first` | batch-1 | 0 | FOLLOW:41 | 41 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 580.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-1:r0` | first | `a1-pair-23-3-first` | batch-1 | 0 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-1:r0` | llm | `a1-pair-23-3-first` | batch-1 | 0 | FOLLOW:41 | 41 | 3 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 577.2 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-1:r0` | first | `a0-original-control` | batch-1 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-1:r0` | llm | `a0-original-control` | batch-1 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 575.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-1:r1` | first | `a4-pair-34-41-first` | batch-1 | 1 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-1:r1` | llm | `a4-pair-34-41-first` | batch-1 | 1 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 575.9 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-1:r1` | first | `a3-pair-34-3-first` | batch-1 | 1 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-1:r1` | llm | `a3-pair-34-3-first` | batch-1 | 1 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 577.7 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-1:r1` | first | `a2-pair-23-41-first` | batch-1 | 1 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-1:r1` | llm | `a2-pair-23-41-first` | batch-1 | 1 | FOLLOW:41 | 41 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 576.5 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-1:r1` | first | `a1-pair-23-3-first` | batch-1 | 1 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-1:r1` | llm | `a1-pair-23-3-first` | batch-1 | 1 | FOLLOW:41 | 41 | 3 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 575.6 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-1:r1` | first | `a0-original-control` | batch-1 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-1:r1` | llm | `a0-original-control` | batch-1 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 579.9 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-1:r2` | first | `a4-pair-34-41-first` | batch-1 | 2 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first:batch-1:r2` | llm | `a4-pair-34-41-first` | batch-1 | 2 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a4-pair-34-41-first` | False | False | ok | 576.3 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-1:r2` | first | `a3-pair-34-3-first` | batch-1 | 2 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first:batch-1:r2` | llm | `a3-pair-34-3-first` | batch-1 | 2 | FOLLOW:43 | 43 | 5 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a3-pair-34-3-first` | False | False | ok | 578.4 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-1:r2` | first | `a2-pair-23-41-first` | batch-1 | 2 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first:batch-1:r2` | llm | `a2-pair-23-41-first` | batch-1 | 2 | FOLLOW:41 | 41 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a2-pair-23-41-first` | False | False | ok | 577.1 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-1:r2` | first | `a1-pair-23-3-first` | batch-1 | 2 | FOLLOW:8 | 8 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first:batch-1:r2` | llm | `a1-pair-23-3-first` | batch-1 | 2 | FOLLOW:41 | 41 | 3 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a1-pair-23-3-first` | False | False | ok | 576.9 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-1:r2` | first | `a0-original-control` | batch-1 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control:batch-1:r2` | llm | `a0-original-control` | batch-1 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:candidate-relative-position:a0-original-control` | False | False | ok | 577.3 |

## Results

### Metrics

- scheduled 60 / valid 60 / errors 0 / missing 0
- independent_question_count = 1
- this is a diagnostic over frozen single-step gate inputs, not a full-loop retrieval result, not an internal-reasoning trace

**Per-policy coverage and accuracy**

| policy | scheduled | valid | errors | missing | selection accuracy | decision accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| first | 30 | 30 | 0 | 0 | 0.0000 (0/30) | 0.0000 (0/30) |
| llm | 30 | 30 | 0 | 0 | 0.0000 (0/30) | 0.0000 (0/30) |

**Action counts per arm, batch and repeat**

| policy | arm | batch | repeat | scheduled | valid | errors | missing | actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first | `a0-original-control` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:41': 6} |
| first | `a1-pair-23-3-first` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:8': 6} |
| first | `a2-pair-23-41-first` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:8': 6} |
| first | `a3-pair-34-3-first` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:8': 6} |
| first | `a4-pair-34-41-first` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:8': 6} |
| llm | `a0-original-control` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:41': 6} |
| llm | `a1-pair-23-3-first` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:41': 6} |
| llm | `a2-pair-23-41-first` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:41': 6} |
| llm | `a3-pair-34-3-first` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:43': 6} |
| llm | `a4-pair-34-41-first` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:43': 6} |
| first | `a0-original-control` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `a0-original-control` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `a0-original-control` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `a0-original-control` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `a0-original-control` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `a0-original-control` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `a1-pair-23-3-first` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a1-pair-23-3-first` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a1-pair-23-3-first` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a1-pair-23-3-first` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a1-pair-23-3-first` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a1-pair-23-3-first` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a2-pair-23-41-first` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a2-pair-23-41-first` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a2-pair-23-41-first` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a2-pair-23-41-first` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a2-pair-23-41-first` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a2-pair-23-41-first` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a3-pair-34-3-first` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a3-pair-34-3-first` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a3-pair-34-3-first` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a3-pair-34-3-first` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a3-pair-34-3-first` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a3-pair-34-3-first` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a4-pair-34-41-first` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a4-pair-34-41-first` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a4-pair-34-41-first` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a4-pair-34-41-first` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a4-pair-34-41-first` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| first | `a4-pair-34-41-first` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:8': 1} |
| llm | `a0-original-control` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a0-original-control` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a0-original-control` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a0-original-control` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a0-original-control` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a0-original-control` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a1-pair-23-3-first` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a1-pair-23-3-first` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a1-pair-23-3-first` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a1-pair-23-3-first` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a1-pair-23-3-first` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a1-pair-23-3-first` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a2-pair-23-41-first` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a2-pair-23-41-first` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a2-pair-23-41-first` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a2-pair-23-41-first` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a2-pair-23-41-first` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a2-pair-23-41-first` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `a3-pair-34-3-first` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a3-pair-34-3-first` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a3-pair-34-3-first` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a3-pair-34-3-first` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a3-pair-34-3-first` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a3-pair-34-3-first` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a4-pair-34-41-first` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a4-pair-34-41-first` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a4-pair-34-41-first` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a4-pair-34-41-first` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a4-pair-34-41-first` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |
| llm | `a4-pair-34-41-first` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:43': 1} |

**Accuracy per arm and case**

| policy | arm | case | scheduled | valid | errors | missing | selection | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first | `a0-original-control` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| first | `a1-pair-23-3-first` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| first | `a2-pair-23-41-first` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| first | `a3-pair-34-3-first` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| first | `a4-pair-34-41-first` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| llm | `a0-original-control` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| llm | `a1-pair-23-3-first` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| llm | `a2-pair-23-41-first` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| llm | `a3-pair-34-3-first` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| llm | `a4-pair-34-41-first` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |

**Within-exact-input repeat consistency**

| policy | arm | batch | scheduled | valid | pairs | matching | consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first | `a0-original-control` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `a0-original-control` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `a1-pair-23-3-first` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `a1-pair-23-3-first` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `a2-pair-23-41-first` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `a2-pair-23-41-first` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `a3-pair-34-3-first` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `a3-pair-34-3-first` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `a4-pair-34-41-first` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `a4-pair-34-41-first` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `a0-original-control` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `a0-original-control` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `a1-pair-23-3-first` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `a1-pair-23-3-first` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `a2-pair-23-41-first` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `a2-pair-23-41-first` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `a3-pair-34-3-first` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `a3-pair-34-3-first` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `a4-pair-34-41-first` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `a4-pair-34-41-first` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |

**Batch-to-batch agreement**

| policy | arm | repeat | scheduled | valid | agreement | switches |
| --- | --- | --- | --- | --- | --- | --- |
| first | `a0-original-control` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a0-original-control` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a0-original-control` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a1-pair-23-3-first` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a1-pair-23-3-first` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a1-pair-23-3-first` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a2-pair-23-41-first` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a2-pair-23-41-first` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a2-pair-23-41-first` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a3-pair-34-3-first` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a3-pair-34-3-first` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a3-pair-34-3-first` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a4-pair-34-41-first` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a4-pair-34-41-first` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `a4-pair-34-41-first` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a0-original-control` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a0-original-control` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a0-original-control` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a1-pair-23-3-first` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a1-pair-23-3-first` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a1-pair-23-3-first` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a2-pair-23-41-first` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a2-pair-23-41-first` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a2-pair-23-41-first` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a3-pair-34-3-first` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a3-pair-34-3-first` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a3-pair-34-3-first` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a4-pair-34-41-first` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a4-pair-34-41-first` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `a4-pair-34-41-first` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |

**Intervention A slices**

| policy | arm | article 3 | article 41 | earlier-of-pair | pair selections | STOP | follows outside the pair | pair-slot positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first | `a0-original-control` | 0 | 6 | 1.0000 (6/6) | 6 | 0 | 0 | {'1': 6} |
| first | `a1-pair-23-3-first` | 0 | 0 | n/a (0/0) | 0 | 0 | 6 | {} |
| first | `a2-pair-23-41-first` | 0 | 0 | n/a (0/0) | 0 | 0 | 6 | {} |
| first | `a3-pair-34-3-first` | 0 | 0 | n/a (0/0) | 0 | 0 | 6 | {} |
| first | `a4-pair-34-41-first` | 0 | 0 | n/a (0/0) | 0 | 0 | 6 | {} |
| llm | `a0-original-control` | 0 | 6 | 1.0000 (6/6) | 6 | 0 | 0 | {'1': 6} |
| llm | `a1-pair-23-3-first` | 0 | 6 | 0.0000 (0/6) | 6 | 0 | 0 | {'3': 6} |
| llm | `a2-pair-23-41-first` | 0 | 6 | 1.0000 (6/6) | 6 | 0 | 0 | {'2': 6} |
| llm | `a3-pair-34-3-first` | 0 | 0 | n/a (0/0) | 0 | 0 | 6 | {} |
| llm | `a4-pair-34-41-first` | 0 | 0 | n/a (0/0) | 0 | 0 | 6 | {} |

**Matched-swap action transitions**

| policy | comparison | scheduled | valid | incomplete | agreement | switches | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first | `a1-pair-23-3-first<->a2-pair-23-41-first` | 6 | 6 | 0 | 1.0000 (6/6) | 0 | rows ['FOLLOW:8']: {'FOLLOW:8': {'FOLLOW:8': 6}} |
| llm | `a1-pair-23-3-first<->a2-pair-23-41-first` | 6 | 6 | 0 | 1.0000 (6/6) | 0 | rows ['FOLLOW:41']: {'FOLLOW:41': {'FOLLOW:41': 6}} |
| first | `a3-pair-34-3-first<->a4-pair-34-41-first` | 6 | 6 | 0 | 1.0000 (6/6) | 0 | rows ['FOLLOW:8']: {'FOLLOW:8': {'FOLLOW:8': 6}} |
| llm | `a3-pair-34-3-first<->a4-pair-34-41-first` | 6 | 6 | 0 | 1.0000 (6/6) | 0 | rows ['FOLLOW:43']: {'FOLLOW:43': {'FOLLOW:43': 6}} |

## Exclusions

- `corpus_cross_references:hop0:q-6b86b273ff34`: not_selected_case
- `corpus_cross_references:hop0:q-d4735e3a265e`: not_selected_case
- `corpus_cross_references:hop0:q-4e07408562be`: not_selected_case
- `corpus_cross_references:hop0:q-4b227777d4dd`: not_selected_case
- `corpus_cross_references:hop0:q-e7f6c011776e`: not_selected_case
- `corpus_cross_references:hop0:q-7902699be42c`: not_selected_case
- `corpus_cross_references:hop0:q-2c624232cdd2`: not_selected_case
