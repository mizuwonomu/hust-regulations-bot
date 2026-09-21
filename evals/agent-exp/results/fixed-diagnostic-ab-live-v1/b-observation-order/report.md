# Fixed diagnostic report: f0958dace891

- suite: `fixed-diagnostic-b-v1`
- diagnostic kind: `observation-block-order`
- case: `corpus_cross_references:hop0:q-ef2d127de37b`
- status: `complete`
- policies: ['first', 'llm']
- source revision: `35bd7b1`
- execution revision: `d863d1f79607e3da042fe1cc2253b51c83bf7f09`
- independent_question_count = 1
- started at: 2026-09-16T15:50:55.247451+00:00
- ended at: 2026-09-17T07:31:44.550084+00:00

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
| `b0-original-control` | [41, 3, 40, 8, 43] | [1, 2] | 41-3 | ['b42-foundation-timing', 'b42-university-defense'] | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` |
| `b1-reversed-blocks` | [41, 3, 40, 8, 43] | [1, 2] | 41-3 | ['b42-university-defense', 'b42-foundation-timing'] | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` |

| batch | repeat | trial | arm |
| --- | --- | --- | --- |
| batch-0 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-0:r0` | `b0-original-control` |
| batch-0 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-0:r0` | `b1-reversed-blocks` |
| batch-0 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-0:r1` | `b0-original-control` |
| batch-0 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-0:r1` | `b1-reversed-blocks` |
| batch-0 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-0:r2` | `b0-original-control` |
| batch-0 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-0:r2` | `b1-reversed-blocks` |
| batch-1 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-1:r0` | `b1-reversed-blocks` |
| batch-1 | 0 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-1:r0` | `b0-original-control` |
| batch-1 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-1:r1` | `b1-reversed-blocks` |
| batch-1 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-1:r1` | `b0-original-control` |
| batch-1 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-1:r2` | `b1-reversed-blocks` |
| batch-1 | 2 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-1:r2` | `b0-original-control` |

## Observation evidence

### Observation transformation evidence

#### Arm `b0-original-control`

#### Original observation

```text
Các Điều đã thu thập:
[Context 42] ĐÀO TẠO TIẾN SĨ - Điều 42. Đánh giá luận án tiến sĩ
- a) NCS được bảo vệ luận án cấp cơ sở nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 1 Điều 41 Quy chế này và hoàn thành hồ sơ đăng ký bảo vệ cấp cơ sở trong thời hạn
- đủ 90 ngày tính đến thời điểm kết thúc thời gian học tập chương trình tiến sĩ quy định tại khoản 3 Điều 3 Quy chế này. Trong trường hợp không đủ thời gian 90 ngày, NCS phải làm thủ tục xin gia hạn thời gian học tập theo quy định.
- a) NCS được bảo vệ luận án cấp Đại học nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 2 Điều 40 Quy chế này và hoàn thành hồ sơ đề nghị đánh giá luận án ở hội đồng đánh giá luận án cấp Đại học.
[Context 45] ĐÀO TẠO TIẾN SĨ - Điều 45. Những thay đổi trong quá trình đào tạo
- 3. Việc chuyển cơ sở đào tạo khác theo khoản 3 Điều 8 Quy chế này.
- a) Hết thời gian theo quy định, NCS không hoàn thành CTĐT hoặc luận án của NCS không được hội đồng đánh giá luận án cấp Đại học thông qua (bao gồm cả trường hợp cho phép đánh giá lại theo quy định tại Điều 43 Quy chế này).
```

#### Transformed observation (b0-original-control)

```text
Các Điều đã thu thập:
[Context 42] ĐÀO TẠO TIẾN SĨ - Điều 42. Đánh giá luận án tiến sĩ
- a) NCS được bảo vệ luận án cấp cơ sở nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 1 Điều 41 Quy chế này và hoàn thành hồ sơ đăng ký bảo vệ cấp cơ sở trong thời hạn
- đủ 90 ngày tính đến thời điểm kết thúc thời gian học tập chương trình tiến sĩ quy định tại khoản 3 Điều 3 Quy chế này. Trong trường hợp không đủ thời gian 90 ngày, NCS phải làm thủ tục xin gia hạn thời gian học tập theo quy định.
- a) NCS được bảo vệ luận án cấp Đại học nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 2 Điều 40 Quy chế này và hoàn thành hồ sơ đề nghị đánh giá luận án ở hội đồng đánh giá luận án cấp Đại học.
[Context 45] ĐÀO TẠO TIẾN SĨ - Điều 45. Những thay đổi trong quá trình đào tạo
- 3. Việc chuyển cơ sở đào tạo khác theo khoản 3 Điều 8 Quy chế này.
- a) Hết thời gian theo quy định, NCS không hoàn thành CTĐT hoặc luận án của NCS không được hội đồng đánh giá luận án cấp Đại học thông qua (bao gồm cả trường hợp cho phép đánh giá lại theo quy định tại Điều 43 Quy chế này).
```

**Block movement**

| block | before | after | before bytes | after bytes | referenced candidates |
| --- | --- | --- | --- | --- | --- |
| `b42-foundation-timing` | 1 | 1 | [114, 659] | [114, 659] | [41, 3] |
| `b42-university-defense` | 2 | 2 | [659, 946] | [659, 946] | [40] |

**Line-origin map**

| output line | original line | segment | line sha256 |
| --- | --- | --- | --- |
| 1 | 1 | `obs-preamble` | `b75b542ccd560676` |
| 2 | 2 | `obs-preamble` | `ecf8728d14d9e26e` |
| 3 | 3 | `b42-foundation-timing` | `ecf9c57373604857` |
| 4 | 4 | `b42-foundation-timing` | `fa30ddecb4b25dd4` |
| 5 | 5 | `b42-university-defense` | `7dea2278ac343e0a` |
| 6 | 6 | `ctx45-section` | `4ba1f7eb718ccc02` |
| 7 | 7 | `ctx45-section` | `fc1178d23ea8b796` |
| 8 | 8 | `ctx45-section` | `dd9e3816977ec0d7` |

**Unchanged sections**

| section | source bytes | output bytes | sha256 | referenced candidates |
| --- | --- | --- | --- | --- |
| `obs-preamble` | [0, 114] | [0, 114] | `822f578bfa2df76d` | [] |
| `ctx45-section` | [946, 1427] | [946, 1427] | `d7958e277a3f1eca` | [8, 43] |

**Article 41/3 page-split group**

- group `b42-foundation-timing-page-split` stays inside block `b42-foundation-timing`
- original lines [3, 4] -> output lines [3, 4]
- line hashes ['ecf9c573736048571de62e435bec736eed9ac4233e553b55c49385b4f19ec88a', 'fa30ddecb4b25dd46829532022274693909a930c8fe9a1dc537f12647d3e005b']

Declared changed fields: []; verified unchanged fields: ['question', 'observation', 'grammar_candidates', 'presented_candidates', 'unchanged_sections']

#### Arm `b1-reversed-blocks`

#### Original observation

```text
Các Điều đã thu thập:
[Context 42] ĐÀO TẠO TIẾN SĨ - Điều 42. Đánh giá luận án tiến sĩ
- a) NCS được bảo vệ luận án cấp cơ sở nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 1 Điều 41 Quy chế này và hoàn thành hồ sơ đăng ký bảo vệ cấp cơ sở trong thời hạn
- đủ 90 ngày tính đến thời điểm kết thúc thời gian học tập chương trình tiến sĩ quy định tại khoản 3 Điều 3 Quy chế này. Trong trường hợp không đủ thời gian 90 ngày, NCS phải làm thủ tục xin gia hạn thời gian học tập theo quy định.
- a) NCS được bảo vệ luận án cấp Đại học nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 2 Điều 40 Quy chế này và hoàn thành hồ sơ đề nghị đánh giá luận án ở hội đồng đánh giá luận án cấp Đại học.
[Context 45] ĐÀO TẠO TIẾN SĨ - Điều 45. Những thay đổi trong quá trình đào tạo
- 3. Việc chuyển cơ sở đào tạo khác theo khoản 3 Điều 8 Quy chế này.
- a) Hết thời gian theo quy định, NCS không hoàn thành CTĐT hoặc luận án của NCS không được hội đồng đánh giá luận án cấp Đại học thông qua (bao gồm cả trường hợp cho phép đánh giá lại theo quy định tại Điều 43 Quy chế này).
```

#### Transformed observation (b1-reversed-blocks)

```text
Các Điều đã thu thập:
[Context 42] ĐÀO TẠO TIẾN SĨ - Điều 42. Đánh giá luận án tiến sĩ
- a) NCS được bảo vệ luận án cấp Đại học nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 2 Điều 40 Quy chế này và hoàn thành hồ sơ đề nghị đánh giá luận án ở hội đồng đánh giá luận án cấp Đại học.
- a) NCS được bảo vệ luận án cấp cơ sở nếu đáp ứng đầy đủ các điều kiện quy định tại khoản 1 Điều 41 Quy chế này và hoàn thành hồ sơ đăng ký bảo vệ cấp cơ sở trong thời hạn
- đủ 90 ngày tính đến thời điểm kết thúc thời gian học tập chương trình tiến sĩ quy định tại khoản 3 Điều 3 Quy chế này. Trong trường hợp không đủ thời gian 90 ngày, NCS phải làm thủ tục xin gia hạn thời gian học tập theo quy định.
[Context 45] ĐÀO TẠO TIẾN SĨ - Điều 45. Những thay đổi trong quá trình đào tạo
- 3. Việc chuyển cơ sở đào tạo khác theo khoản 3 Điều 8 Quy chế này.
- a) Hết thời gian theo quy định, NCS không hoàn thành CTĐT hoặc luận án của NCS không được hội đồng đánh giá luận án cấp Đại học thông qua (bao gồm cả trường hợp cho phép đánh giá lại theo quy định tại Điều 43 Quy chế này).
```

**Block movement**

| block | before | after | before bytes | after bytes | referenced candidates |
| --- | --- | --- | --- | --- | --- |
| `b42-foundation-timing` | 1 | 2 | [114, 659] | [401, 946] | [41, 3] |
| `b42-university-defense` | 2 | 1 | [659, 946] | [114, 401] | [40] |

**Line-origin map**

| output line | original line | segment | line sha256 |
| --- | --- | --- | --- |
| 1 | 1 | `obs-preamble` | `b75b542ccd560676` |
| 2 | 2 | `obs-preamble` | `ecf8728d14d9e26e` |
| 3 | 5 | `b42-university-defense` | `7dea2278ac343e0a` |
| 4 | 3 | `b42-foundation-timing` | `ecf9c57373604857` |
| 5 | 4 | `b42-foundation-timing` | `fa30ddecb4b25dd4` |
| 6 | 6 | `ctx45-section` | `4ba1f7eb718ccc02` |
| 7 | 7 | `ctx45-section` | `fc1178d23ea8b796` |
| 8 | 8 | `ctx45-section` | `dd9e3816977ec0d7` |

**Unchanged sections**

| section | source bytes | output bytes | sha256 | referenced candidates |
| --- | --- | --- | --- | --- |
| `obs-preamble` | [0, 114] | [0, 114] | `822f578bfa2df76d` | [] |
| `ctx45-section` | [946, 1427] | [946, 1427] | `d7958e277a3f1eca` | [8, 43] |

**Article 41/3 page-split group**

- group `b42-foundation-timing-page-split` stays inside block `b42-foundation-timing`
- original lines [3, 4] -> output lines [4, 5]
- line hashes ['ecf9c573736048571de62e435bec736eed9ac4233e553b55c49385b4f19ec88a', 'fa30ddecb4b25dd46829532022274693909a930c8fe9a1dc537f12647d3e005b']

Declared changed fields: ['observation_block_order']; verified unchanged fields: ['question', 'observation', 'grammar_candidates', 'presented_candidates', 'unchanged_sections']

## Decisions

### Decisions

| trial | policy | arm | batch | repeat | action | selected article | selected position | input trace | selection correct | decision correct | outcome | latency ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-0:r0` | first | `b0-original-control` | batch-0 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-0:r0` | llm | `b0-original-control` | batch-0 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 317.9 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-0:r0` | first | `b1-reversed-blocks` | batch-0 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-0:r0` | llm | `b1-reversed-blocks` | batch-0 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 1374.8 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-0:r1` | first | `b0-original-control` | batch-0 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-0:r1` | llm | `b0-original-control` | batch-0 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 578.8 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-0:r1` | first | `b1-reversed-blocks` | batch-0 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-0:r1` | llm | `b1-reversed-blocks` | batch-0 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 580.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-0:r2` | first | `b0-original-control` | batch-0 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-0:r2` | llm | `b0-original-control` | batch-0 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 577.8 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-0:r2` | first | `b1-reversed-blocks` | batch-0 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-0:r2` | llm | `b1-reversed-blocks` | batch-0 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 579.5 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-1:r0` | first | `b1-reversed-blocks` | batch-1 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-1:r0` | llm | `b1-reversed-blocks` | batch-1 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 317.5 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-1:r0` | first | `b0-original-control` | batch-1 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-1:r0` | llm | `b0-original-control` | batch-1 | 0 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 1371.7 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-1:r1` | first | `b1-reversed-blocks` | batch-1 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-1:r1` | llm | `b1-reversed-blocks` | batch-1 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 581.6 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-1:r1` | first | `b0-original-control` | batch-1 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-1:r1` | llm | `b0-original-control` | batch-1 | 1 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 585.2 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-1:r2` | first | `b1-reversed-blocks` | batch-1 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks:batch-1:r2` | llm | `b1-reversed-blocks` | batch-1 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks` | False | False | ok | 586.6 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-1:r2` | first | `b0-original-control` | batch-1 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 0.0 |
| `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control:batch-1:r2` | llm | `b0-original-control` | batch-1 | 2 | FOLLOW:41 | 41 | 1 | `corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control` | False | False | ok | 584.6 |

## Results

### Metrics

- scheduled 24 / valid 24 / errors 0 / missing 0
- independent_question_count = 1
- this is a diagnostic over frozen single-step gate inputs, not a full-loop retrieval result, not an internal-reasoning trace

**Per-policy coverage and accuracy**

| policy | scheduled | valid | errors | missing | selection accuracy | decision accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| first | 12 | 12 | 0 | 0 | 0.0000 (0/12) | 0.0000 (0/12) |
| llm | 12 | 12 | 0 | 0 | 0.0000 (0/12) | 0.0000 (0/12) |

**Action counts per arm, batch and repeat**

| policy | arm | batch | repeat | scheduled | valid | errors | missing | actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first | `b0-original-control` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:41': 6} |
| first | `b1-reversed-blocks` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:41': 6} |
| llm | `b0-original-control` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:41': 6} |
| llm | `b1-reversed-blocks` | pooled | pooled | 6 | 6 | 0 | 0 | {'FOLLOW:41': 6} |
| first | `b0-original-control` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b0-original-control` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b0-original-control` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b0-original-control` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b0-original-control` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b0-original-control` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b1-reversed-blocks` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b1-reversed-blocks` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b1-reversed-blocks` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b1-reversed-blocks` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b1-reversed-blocks` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| first | `b1-reversed-blocks` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b0-original-control` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b0-original-control` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b0-original-control` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b0-original-control` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b0-original-control` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b0-original-control` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b1-reversed-blocks` | batch-0 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b1-reversed-blocks` | batch-0 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b1-reversed-blocks` | batch-0 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b1-reversed-blocks` | batch-1 | 0 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b1-reversed-blocks` | batch-1 | 1 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |
| llm | `b1-reversed-blocks` | batch-1 | 2 | 1 | 1 | 0 | 0 | {'FOLLOW:41': 1} |

**Accuracy per arm and case**

| policy | arm | case | scheduled | valid | errors | missing | selection | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first | `b0-original-control` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| first | `b1-reversed-blocks` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| llm | `b0-original-control` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |
| llm | `b1-reversed-blocks` | `corpus_cross_references:hop0:q-ef2d127de37b` | 6 | 6 | 0 | 0 | 0.0000 (0/6) | 0.0000 (0/6) |

**Within-exact-input repeat consistency**

| policy | arm | batch | scheduled | valid | pairs | matching | consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first | `b0-original-control` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `b0-original-control` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `b1-reversed-blocks` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| first | `b1-reversed-blocks` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `b0-original-control` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `b0-original-control` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `b1-reversed-blocks` | batch-0 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |
| llm | `b1-reversed-blocks` | batch-1 | 3 | 3 | 3 | 3 | 1.0000 (3/3) |

**Batch-to-batch agreement**

| policy | arm | repeat | scheduled | valid | agreement | switches |
| --- | --- | --- | --- | --- | --- | --- |
| first | `b0-original-control` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `b0-original-control` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `b0-original-control` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `b1-reversed-blocks` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `b1-reversed-blocks` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| first | `b1-reversed-blocks` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `b0-original-control` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `b0-original-control` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `b0-original-control` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `b1-reversed-blocks` | 0 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `b1-reversed-blocks` | 1 | 1 | 1 | 1.0000 (1/1) | 0 |
| llm | `b1-reversed-blocks` | 2 | 1 | 1 | 1.0000 (1/1) | 0 |

**Intervention B slices**

**Original-versus-reversed transitions**

| policy | comparison | scheduled | valid | incomplete | agreement | switches | transition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first | `b0-original-control<->b1-reversed-blocks` | 6 | 6 | 0 | 1.0000 (6/6) | 0 | rows ['FOLLOW:41']: {'FOLLOW:41': {'FOLLOW:41': 6}} |
| llm | `b0-original-control<->b1-reversed-blocks` | 6 | 6 | 0 | 1.0000 (6/6) | 0 | rows ['FOLLOW:41']: {'FOLLOW:41': {'FOLLOW:41': 6}} |

- Articles 3 and 41 share b42-foundation-timing, so intervention B cannot distinguish the order of their mention inside that atomic sentence

- transformation trace identity: {'b0-original-control': 'corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b0-original-control', 'b1-reversed-blocks': 'corpus_cross_references:hop0:q-ef2d127de37b:observation-block-order:b1-reversed-blocks'}

**Selected candidate supporting evidence**

| policy | arm | batch | repeat | selected article | block ids | 1-based block positions |
| --- | --- | --- | --- | --- | --- | --- |
| first | `b0-original-control` | batch-0 | 0 | 41 | ['b42-foundation-timing'] | [2] |
| first | `b1-reversed-blocks` | batch-0 | 0 | 41 | ['b42-foundation-timing'] | [3] |
| first | `b0-original-control` | batch-0 | 1 | 41 | ['b42-foundation-timing'] | [2] |
| first | `b1-reversed-blocks` | batch-0 | 1 | 41 | ['b42-foundation-timing'] | [3] |
| first | `b0-original-control` | batch-0 | 2 | 41 | ['b42-foundation-timing'] | [2] |
| first | `b1-reversed-blocks` | batch-0 | 2 | 41 | ['b42-foundation-timing'] | [3] |
| first | `b1-reversed-blocks` | batch-1 | 0 | 41 | ['b42-foundation-timing'] | [3] |
| first | `b0-original-control` | batch-1 | 0 | 41 | ['b42-foundation-timing'] | [2] |
| first | `b1-reversed-blocks` | batch-1 | 1 | 41 | ['b42-foundation-timing'] | [3] |
| first | `b0-original-control` | batch-1 | 1 | 41 | ['b42-foundation-timing'] | [2] |
| first | `b1-reversed-blocks` | batch-1 | 2 | 41 | ['b42-foundation-timing'] | [3] |
| first | `b0-original-control` | batch-1 | 2 | 41 | ['b42-foundation-timing'] | [2] |
| llm | `b0-original-control` | batch-0 | 0 | 41 | ['b42-foundation-timing'] | [2] |
| llm | `b1-reversed-blocks` | batch-0 | 0 | 41 | ['b42-foundation-timing'] | [3] |
| llm | `b0-original-control` | batch-0 | 1 | 41 | ['b42-foundation-timing'] | [2] |
| llm | `b1-reversed-blocks` | batch-0 | 1 | 41 | ['b42-foundation-timing'] | [3] |
| llm | `b0-original-control` | batch-0 | 2 | 41 | ['b42-foundation-timing'] | [2] |
| llm | `b1-reversed-blocks` | batch-0 | 2 | 41 | ['b42-foundation-timing'] | [3] |
| llm | `b1-reversed-blocks` | batch-1 | 0 | 41 | ['b42-foundation-timing'] | [3] |
| llm | `b0-original-control` | batch-1 | 0 | 41 | ['b42-foundation-timing'] | [2] |
| llm | `b1-reversed-blocks` | batch-1 | 1 | 41 | ['b42-foundation-timing'] | [3] |
| llm | `b0-original-control` | batch-1 | 1 | 41 | ['b42-foundation-timing'] | [2] |
| llm | `b1-reversed-blocks` | batch-1 | 2 | 41 | ['b42-foundation-timing'] | [3] |
| llm | `b0-original-control` | batch-1 | 2 | 41 | ['b42-foundation-timing'] | [2] |

## Exclusions

- `corpus_cross_references:hop0:q-6b86b273ff34`: not_selected_case
- `corpus_cross_references:hop0:q-d4735e3a265e`: not_selected_case
- `corpus_cross_references:hop0:q-4e07408562be`: not_selected_case
- `corpus_cross_references:hop0:q-4b227777d4dd`: not_selected_case
- `corpus_cross_references:hop0:q-e7f6c011776e`: not_selected_case
- `corpus_cross_references:hop0:q-7902699be42c`: not_selected_case
- `corpus_cross_references:hop0:q-2c624232cdd2`: not_selected_case
