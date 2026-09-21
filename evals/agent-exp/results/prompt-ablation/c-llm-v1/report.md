# Defense few-shot ablation C

Status: complete

Baseline: 14 messages, 5 FOLLOW / 1 STOP; ablated: 8 messages, 2 FOLLOW / 1 STOP

| Policy | Phase | Q | Arm | Baseline actions | Ablated actions | Valid pairs | Switches |
|---|---|---|---|---|---|---|---|
| first | c1-q5-candidate-interaction | 5 | a0-original-control | {'FOLLOW:41': 6} | {'FOLLOW:41': 6} | 6/6 | 0 |
| first | c1-q5-candidate-interaction | 5 | a1-pair-23-3-first | {'FOLLOW:8': 6} | {'FOLLOW:8': 6} | 6/6 | 0 |
| first | c1-q5-candidate-interaction | 5 | a2-pair-23-41-first | {'FOLLOW:8': 6} | {'FOLLOW:8': 6} | 6/6 | 0 |
| first | c1-q5-candidate-interaction | 5 | a3-pair-34-3-first | {'FOLLOW:8': 6} | {'FOLLOW:8': 6} | 6/6 | 0 |
| first | c1-q5-candidate-interaction | 5 | a4-pair-34-41-first | {'FOLLOW:8': 6} | {'FOLLOW:8': 6} | 6/6 | 0 |
| first | c2-eligible-corpus-guard | 1 | original | {'FOLLOW:19': 6} | {'FOLLOW:19': 6} | 6/6 | 0 |
| first | c2-eligible-corpus-guard | 3 | original | {'FOLLOW:20': 6} | {'FOLLOW:20': 6} | 6/6 | 0 |
| first | c2-eligible-corpus-guard | 4 | original | {'FOLLOW:3': 6} | {'FOLLOW:3': 6} | 6/6 | 0 |
| first | c2-eligible-corpus-guard | 5 | original | {'FOLLOW:41': 6} | {'FOLLOW:41': 6} | 6/6 | 0 |
| first | c2-eligible-corpus-guard | 6 | original | {'FOLLOW:3': 6} | {'FOLLOW:3': 6} | 6/6 | 0 |
| first | c2-eligible-corpus-guard | 7 | original | {'FOLLOW:12': 6} | {'FOLLOW:12': 6} | 6/6 | 0 |
| llm | c1-q5-candidate-interaction | 5 | a0-original-control | {'FOLLOW:41': 6} | {'FOLLOW:41': 6} | 6/6 | 0 |
| llm | c1-q5-candidate-interaction | 5 | a1-pair-23-3-first | {'FOLLOW:41': 6} | {'FOLLOW:41': 6} | 6/6 | 0 |
| llm | c1-q5-candidate-interaction | 5 | a2-pair-23-41-first | {'FOLLOW:41': 6} | {'FOLLOW:41': 6} | 6/6 | 0 |
| llm | c1-q5-candidate-interaction | 5 | a3-pair-34-3-first | {'FOLLOW:43': 6} | {'FOLLOW:3': 6} | 6/6 | 6 |
| llm | c1-q5-candidate-interaction | 5 | a4-pair-34-41-first | {'FOLLOW:43': 6} | {'FOLLOW:41': 6} | 6/6 | 6 |
| llm | c2-eligible-corpus-guard | 1 | original | {'FOLLOW:19': 6} | {'FOLLOW:19': 6} | 6/6 | 0 |
| llm | c2-eligible-corpus-guard | 3 | original | {'FOLLOW:20': 6} | {'FOLLOW:20': 6} | 6/6 | 0 |
| llm | c2-eligible-corpus-guard | 4 | original | {'FOLLOW:3': 6} | {'FOLLOW:3': 6} | 6/6 | 0 |
| llm | c2-eligible-corpus-guard | 5 | original | {'FOLLOW:41': 6} | {'FOLLOW:41': 6} | 6/6 | 0 |
| llm | c2-eligible-corpus-guard | 6 | original | {'FOLLOW:3': 6} | {'FOLLOW:3': 6} | 6/6 | 0 |
| llm | c2-eligible-corpus-guard | 7 | original | {'FOLLOW:12': 6} | {'FOLLOW:12': 6} | 6/6 | 0 |

## Coverage

| Policy | Scheduled | Valid | Errors | Missing |
|---|---|---|---|---|
| first | 132 | 132 | 0 | 0 |
| llm | 132 | 132 | 0 | 0 |

## Repeat completeness

| Policy / phase / arm / variant | Complete / scheduled groups | Comparable / scheduled pairs | Incomplete groups / pairs | Agreement |
|---|---|---|---|---|
| first / c1-q5-candidate-interaction / a0-original-control / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c1-q5-candidate-interaction / a0-original-control / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c1-q5-candidate-interaction / a1-pair-23-3-first / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c1-q5-candidate-interaction / a1-pair-23-3-first / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c1-q5-candidate-interaction / a2-pair-23-41-first / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c1-q5-candidate-interaction / a2-pair-23-41-first / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c1-q5-candidate-interaction / a3-pair-34-3-first / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c1-q5-candidate-interaction / a3-pair-34-3-first / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c1-q5-candidate-interaction / a4-pair-34-41-first / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c1-q5-candidate-interaction / a4-pair-34-41-first / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| first / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / a0-original-control / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / a0-original-control / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / a1-pair-23-3-first / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / a1-pair-23-3-first / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / a2-pair-23-41-first / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / a2-pair-23-41-first / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / a3-pair-34-3-first / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / a3-pair-34-3-first / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / a4-pair-34-41-first / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / a4-pair-34-41-first / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / baseline-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / original / without-doctoral-defense-v1 | 2/2 | 6/6 | 0 / 0 | {'numerator': 6, 'denominator': 6, 'value': 1.0} |

## Batch agreement

| Policy / phase / Q / arm / variant | Complete / scheduled | Incomplete | Agreement |
|---|---|---|---|
| first / c1-q5-candidate-interaction / 5 / a0-original-control / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c1-q5-candidate-interaction / 5 / a0-original-control / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c1-q5-candidate-interaction / 5 / a1-pair-23-3-first / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c1-q5-candidate-interaction / 5 / a1-pair-23-3-first / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c1-q5-candidate-interaction / 5 / a2-pair-23-41-first / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c1-q5-candidate-interaction / 5 / a2-pair-23-41-first / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c1-q5-candidate-interaction / 5 / a3-pair-34-3-first / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c1-q5-candidate-interaction / 5 / a3-pair-34-3-first / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c1-q5-candidate-interaction / 5 / a4-pair-34-41-first / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c1-q5-candidate-interaction / 5 / a4-pair-34-41-first / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 1 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 1 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 3 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 3 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 4 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 4 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 5 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 5 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 6 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 6 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 7 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| first / c2-eligible-corpus-guard / 7 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / 5 / a0-original-control / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / 5 / a0-original-control / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / 5 / a1-pair-23-3-first / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / 5 / a1-pair-23-3-first / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / 5 / a2-pair-23-41-first / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / 5 / a2-pair-23-41-first / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / 5 / a3-pair-34-3-first / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / 5 / a3-pair-34-3-first / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / 5 / a4-pair-34-41-first / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c1-q5-candidate-interaction / 5 / a4-pair-34-41-first / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 1 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 1 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 3 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 3 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 4 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 4 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 5 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 5 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 6 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 6 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 7 / original / baseline-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |
| llm / c2-eligible-corpus-guard / 7 / original / without-doctoral-defense-v1 | 3/3 | 0 | {'numerator': 3, 'denominator': 3, 'value': 1.0} |

## C2 question macro

| Policy | Variant | Metric | Eligible | Defined | Undefined | Mean |
|---|---|---|---|---|---|---|
| first | baseline-v1 | selection_accuracy | 6 | 4 | 2 | 0.75 |
| first | baseline-v1 | decision_accuracy | 6 | 6 | 0 | 0.5 |
| first | without-doctoral-defense-v1 | selection_accuracy | 6 | 4 | 2 | 0.75 |
| first | without-doctoral-defense-v1 | decision_accuracy | 6 | 6 | 0 | 0.5 |
| llm | baseline-v1 | selection_accuracy | 6 | 4 | 2 | 0.75 |
| llm | baseline-v1 | decision_accuracy | 6 | 6 | 0 | 0.5 |
| llm | without-doctoral-defense-v1 | selection_accuracy | 6 | 4 | 2 | 0.75 |
| llm | without-doctoral-defense-v1 | decision_accuracy | 6 | 6 | 0 | 0.5 |

## A-by-C matched swaps

### first: a1-pair-23-3-first -> a2-pair-23-41-first

baseline-v1: 3-before-41 -> 41-before-3
Complete 6/6; incomplete 0; agreement {'numerator': 6, 'denominator': 6, 'value': 1.0}

| From | To | Count |
|---|---|---|
| FOLLOW:8 | FOLLOW:8 | 6 |
without-doctoral-defense-v1: 3-before-41 -> 41-before-3
Complete 6/6; incomplete 0; agreement {'numerator': 6, 'denominator': 6, 'value': 1.0}

| From | To | Count |
|---|---|---|
| FOLLOW:8 | FOLLOW:8 | 6 |

Decision-accuracy difference-in-differences: 0.0; undefined components: 0
C effects (ablated - baseline): {'3-before-41': 0.0, '41-before-3': 0.0}

| Orientation | Variant | Numerator | Denominator | Value |
|---|---|---|---|---|
| 3-before-41 | baseline-v1 | 0 | 6 | 0.0 |
| 3-before-41 | without-doctoral-defense-v1 | 0 | 6 | 0.0 |
| 41-before-3 | baseline-v1 | 0 | 6 | 0.0 |
| 41-before-3 | without-doctoral-defense-v1 | 0 | 6 | 0.0 |
### first: a3-pair-34-3-first -> a4-pair-34-41-first

baseline-v1: 3-before-41 -> 41-before-3
Complete 6/6; incomplete 0; agreement {'numerator': 6, 'denominator': 6, 'value': 1.0}

| From | To | Count |
|---|---|---|
| FOLLOW:8 | FOLLOW:8 | 6 |
without-doctoral-defense-v1: 3-before-41 -> 41-before-3
Complete 6/6; incomplete 0; agreement {'numerator': 6, 'denominator': 6, 'value': 1.0}

| From | To | Count |
|---|---|---|
| FOLLOW:8 | FOLLOW:8 | 6 |

Decision-accuracy difference-in-differences: 0.0; undefined components: 0
C effects (ablated - baseline): {'3-before-41': 0.0, '41-before-3': 0.0}

| Orientation | Variant | Numerator | Denominator | Value |
|---|---|---|---|---|
| 3-before-41 | baseline-v1 | 0 | 6 | 0.0 |
| 3-before-41 | without-doctoral-defense-v1 | 0 | 6 | 0.0 |
| 41-before-3 | baseline-v1 | 0 | 6 | 0.0 |
| 41-before-3 | without-doctoral-defense-v1 | 0 | 6 | 0.0 |
### llm: a1-pair-23-3-first -> a2-pair-23-41-first

baseline-v1: 3-before-41 -> 41-before-3
Complete 6/6; incomplete 0; agreement {'numerator': 6, 'denominator': 6, 'value': 1.0}

| From | To | Count |
|---|---|---|
| FOLLOW:41 | FOLLOW:41 | 6 |
without-doctoral-defense-v1: 3-before-41 -> 41-before-3
Complete 6/6; incomplete 0; agreement {'numerator': 6, 'denominator': 6, 'value': 1.0}

| From | To | Count |
|---|---|---|
| FOLLOW:41 | FOLLOW:41 | 6 |

Decision-accuracy difference-in-differences: 0.0; undefined components: 0
C effects (ablated - baseline): {'3-before-41': 0.0, '41-before-3': 0.0}

| Orientation | Variant | Numerator | Denominator | Value |
|---|---|---|---|---|
| 3-before-41 | baseline-v1 | 0 | 6 | 0.0 |
| 3-before-41 | without-doctoral-defense-v1 | 0 | 6 | 0.0 |
| 41-before-3 | baseline-v1 | 0 | 6 | 0.0 |
| 41-before-3 | without-doctoral-defense-v1 | 0 | 6 | 0.0 |
### llm: a3-pair-34-3-first -> a4-pair-34-41-first

baseline-v1: 3-before-41 -> 41-before-3
Complete 6/6; incomplete 0; agreement {'numerator': 6, 'denominator': 6, 'value': 1.0}

| From | To | Count |
|---|---|---|
| FOLLOW:43 | FOLLOW:43 | 6 |
without-doctoral-defense-v1: 3-before-41 -> 41-before-3
Complete 6/6; incomplete 0; agreement {'numerator': 0, 'denominator': 6, 'value': 0.0}

| From | To | Count |
|---|---|---|
| FOLLOW:3 | FOLLOW:41 | 6 |

Decision-accuracy difference-in-differences: -1.0; undefined components: 0
C effects (ablated - baseline): {'3-before-41': 1.0, '41-before-3': 0.0}

| Orientation | Variant | Numerator | Denominator | Value |
|---|---|---|---|---|
| 3-before-41 | baseline-v1 | 0 | 6 | 0.0 |
| 3-before-41 | without-doctoral-defense-v1 | 6 | 6 | 1.0 |
| 41-before-3 | baseline-v1 | 0 | 6 | 0.0 |
| 41-before-3 | without-doctoral-defense-v1 | 0 | 6 | 0.0 |

## Trace evidence

Full paired requests: input_traces.jsonl and prompt_traces.jsonl
Source parity and literal example registry: manifest.json and prompt_traces.jsonl

## Detailed measurements

Per-batch accuracy, raw denominators, token usage, repeat consistency, batch agreement, role guards and A-by-C interactions are in summary.json

Removed example-group bundle effect only; not proof of memorization, causality of length, or corpus generalization
