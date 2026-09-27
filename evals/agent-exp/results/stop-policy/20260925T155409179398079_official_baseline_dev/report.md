# STOP policy run report

- Experiment: stop-policy
- Result records: 96

### Provenance

- Run: `df0cd3a5eadd` status `complete`
- Purpose: `evaluation`
- Variant: `baseline` | split `dev` | repeats 3 | policies first, llm
- Cases source: `evals/agent-exp/snapshots/stop_policy/20260925T155409179398079_review_v2.jsonl` sha256 `41dade955bb4865af18ce98ccf694cc8519c2c8108bf1538313840d1b08214f3`
- Snapshot source: `evals/agent-exp/snapshots/stop_policy/20260925T040632312542198_review/seeds.json` sha256 `ababa7a6fd23b7beba15e9b3636c51a235685cfe60b3fd13c24f8b71706a54a5`
- Corpus sha256: `35103d125f1de1698bdb15e2a6968090d0625333960ea446e69aa78612323637`
- Execution revision `492811d58d0877091e77c2b0f1b4a276304cedaf` dirty `True` | 33 executed module hashes
- LLM configuration: alias `citation-agent` base `http://127.0.0.1:8080/v1` temperature 0.0 max tokens 4096 thinking False timeout None retries None
- Provenance limits: served model weights and store contents are not verified here

### Class balance and exclusions

- hop0_follow: 3 eligible case(s)
- hop0_stop: 7 eligible case(s)
- post_follow_follow: 1 eligible case(s)
- post_follow_stop: 5 eligible case(s)
- Complete for the four required classes: True (missing: none)
- Exclusions: fewshot_overlap 4, no_candidates 4
- Replay inputs: 16

### Run completeness

- Totals: scheduled 96 | valid 96 | errors 0 | missing 0
- first: scheduled 48 | valid 48 | errors 0 | missing 0
- llm: scheduled 48 | valid 48 | errors 0 | missing 0
- Errors, missing slots and invalid decisions are never counted as STOP

### Action rates by label (valid outputs only)

| policy | follow accuracy | false stop rate | stop accuracy | over-hop rate | follow valid | stop valid |
| --- | --- | --- | --- | --- | --- | --- |
| first | 1.0000 (12/12) | 0.0000 (0/12) | 0.0000 (0/36) | 1.0000 (36/36) | 12 | 36 |
| llm | 1.0000 (12/12) | 0.0000 (0/12) | 0.0000 (0/36) | 1.0000 (36/36) | 12 | 36 |

| policy | action accuracy | selection accuracy | decision accuracy | follow error | follow missing | stop error | stop missing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first | 0.2500 (12/48) | 0.5000 (6/12) | 0.1250 (6/48) | 0.0000 (0/12) | 0.0000 (0/12) | 0.0000 (0/36) | 0.0000 (0/36) |
| llm | 0.2500 (12/48) | 0.7500 (9/12) | 0.1875 (9/48) | 0.0000 (0/12) | 0.0000 (0/12) | 0.0000 (0/36) | 0.0000 (0/36) |

Scheduled accuracies count every labeled slot; errors and missing slots are incorrect there

### Case-level macro means

| policy | cases | follow accuracy | stop accuracy | action | selection | decision |
| --- | --- | --- | --- | --- | --- | --- |
| first | 16 | 1.0000 (defined 4/16) | 0.0000 (defined 12/16) | 0.2500 (defined 16/16) | 0.5000 (defined 4/16) | 0.1250 (defined 16/16) |
| llm | 16 | 1.0000 (defined 4/16) | 0.0000 (defined 12/16) | 0.2500 (defined 16/16) | 0.7500 (defined 4/16) | 0.1875 (defined 16/16) |

Undefined case values are excluded and reported through the defined/eligible counts above

### Strata

- semantic_group:
  - Chuyển sang Hình thức Đào tạo Vừa làm Vừa học (VLVH) [first]: scheduled 6 | valid 6 | errors 0 | missing 0 | action 0.0000 (0/6) | selection n/a (0/0) | decision 0.0000 (0/6)
  - Cảnh báo Học tập và Hạn chế Khối lượng Tín chỉ [first]: scheduled 12 | valid 12 | errors 0 | missing 0 | action 0.2500 (3/12) | selection 1.0000 (3/3) | decision 0.2500 (3/12)
  - Nghỉ học tạm thời và Học phí [first]: scheduled 6 | valid 6 | errors 0 | missing 0 | action 0.0000 (0/6) | selection n/a (0/0) | decision 0.0000 (0/6)
  - Xếp hạng Tốt nghiệp và Tra cứu Điểm/Tín chỉ Tích lũy [first]: scheduled 6 | valid 6 | errors 0 | missing 0 | action 1.0000 (6/6) | selection 0.5000 (3/6) | decision 0.5000 (3/6)
  - Điều khoản Chuyển tiếp và Mốc Hiệu lực Thi hành theo Khóa [first]: scheduled 18 | valid 18 | errors 0 | missing 0 | action 0.1667 (3/18) | selection 0.0000 (0/3) | decision 0.0000 (0/18)
  - Chuyển sang Hình thức Đào tạo Vừa làm Vừa học (VLVH) [llm]: scheduled 6 | valid 6 | errors 0 | missing 0 | action 0.0000 (0/6) | selection n/a (0/0) | decision 0.0000 (0/6)
  - Cảnh báo Học tập và Hạn chế Khối lượng Tín chỉ [llm]: scheduled 12 | valid 12 | errors 0 | missing 0 | action 0.2500 (3/12) | selection 1.0000 (3/3) | decision 0.2500 (3/12)
  - Nghỉ học tạm thời và Học phí [llm]: scheduled 6 | valid 6 | errors 0 | missing 0 | action 0.0000 (0/6) | selection n/a (0/0) | decision 0.0000 (0/6)
  - Xếp hạng Tốt nghiệp và Tra cứu Điểm/Tín chỉ Tích lũy [llm]: scheduled 6 | valid 6 | errors 0 | missing 0 | action 1.0000 (6/6) | selection 0.5000 (3/6) | decision 0.5000 (3/6)
  - Điều khoản Chuyển tiếp và Mốc Hiệu lực Thi hành theo Khóa [llm]: scheduled 18 | valid 18 | errors 0 | missing 0 | action 0.1667 (3/18) | selection 1.0000 (3/3) | decision 0.1667 (3/18)
- source_hop:
  - hop0 [first]: scheduled 30 | valid 30 | errors 0 | missing 0 | action 0.3000 (9/30) | selection 0.3333 (3/9) | decision 0.1000 (3/30)
  - post_follow [first]: scheduled 18 | valid 18 | errors 0 | missing 0 | action 0.1667 (3/18) | selection 1.0000 (3/3) | decision 0.1667 (3/18)
  - hop0 [llm]: scheduled 30 | valid 30 | errors 0 | missing 0 | action 0.3000 (9/30) | selection 0.6667 (6/9) | decision 0.2000 (6/30)
  - post_follow [llm]: scheduled 18 | valid 18 | errors 0 | missing 0 | action 0.1667 (3/18) | selection 1.0000 (3/3) | decision 0.1667 (3/18)
- split:
  - dev [first]: scheduled 48 | valid 48 | errors 0 | missing 0 | action 0.2500 (12/48) | selection 0.5000 (6/12) | decision 0.1250 (6/48)
  - dev [llm]: scheduled 48 | valid 48 | errors 0 | missing 0 | action 0.2500 (12/48) | selection 0.7500 (9/12) | decision 0.1875 (9/48)

### Paired comparison

- Policies first, llm | scheduled pairs 48 | valid pairs 48 | incomplete pairs 0
- Action agreement: 0.8750 (42/48)
- Decision correctness: first wins 0 | llm wins 3 | ties 45
- Trial ids with different actions: 6

### STOP/FOLLOW trade-off

- first: over-hop rate 1.0000 (36/36) against false stop rate 0.0000 (0/12); report both sides together
- llm: over-hop rate 1.0000 (36/36) against false stop rate 0.0000 (0/12); report both sides together

### Conclusion limits

- These figures describe gate decisions on frozen states only; they do not measure answer quality
- No retrieval benefit, full-loop cost or end-to-end improvement is established here
- Heldout numbers are not evidence of generalization until dev selection is frozen
- A stable decision does not isolate reasoning from few-shot or ordering heuristics
- Fake-client tests validate harness mechanics, never model judgment
