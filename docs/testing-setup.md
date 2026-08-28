# Test target setup — Supabase schema isolation

Runbook này ghi lại cách test suite được trỏ vào **Supabase instance thật** nhưng
cách ly bằng **schema riêng**, thay vì localhost Postgres. Mục tiêu (xem
`docs/superpowers/plans/2026-08-21-read-endpoints-cutover.md`, Task 0): mọi con số
latency trong bộ test concurrency phải mô hình đúng đường network thật — RTT, TLS
handshake, hành vi Supavisor pooler. Chia được hai mối quan tâm: **network realism**
(host/pooler/TLS thật) và **data safety** (chỉ chạm schema test, không bao giờ
`public`).

## Schema test: `regu_test_dsk`

Nằm trên cùng project Supabase với dữ liệu production, chỉ khác schema. `clean_db`
TRUNCATE hai bảng trong schema này trước mỗi test — không bao giờ chạm `public`.

### 1. Tạo schema

```sql
CREATE SCHEMA IF NOT EXISTS regu_test_dsk;
```

### 2. Bảng `conversations` — DDL viết tay, khớp production CHÍNH XÁC

Bảng này thuộc app (không phải LangChain) nên viết tay. PK bắt buộc: nó là target
của `ON CONFLICT` trong cả `claim_conversation` lẫn `insert_title_conversations`.
`user_id varchar(15)` giữ nguyên length limit — cột rộng hơn sẽ âm thầm nhận id
mà production từ chối.

```sql
DROP TABLE IF EXISTS regu_test_dsk.conversations;

CREATE TABLE regu_test_dsk.conversations (
    conversation_id text PRIMARY KEY,
    user_id         varchar(15) NOT NULL,
    title           text,                -- NULL được, claim_conversation cố ý insert NULL
    created_at      timestamptz NOT NULL DEFAULT NOW(),  -- get_user_conversations ORDER BY cột này
    updated_at      timestamptz NOT NULL DEFAULT NOW()   -- insert_title_conversations set tường minh khi conflict
);
```

> **Timezone nuance:** `timestamptz` không lưu offset — chuẩn hoá về UTC khi ghi,
> render theo `TimeZone` của session khi đọc. `ORDER BY created_at DESC` so sánh
> thời điểm tuyệt đối nên không bị ảnh hưởng; nếu production session chạy
> `Asia/Ho_Chi_Minh`, đặt test connection cùng timezone để debug không lạ lẫm.

### 3. Bảng `chat_history` — KHÔNG viết tay, dùng chính wrapper LangChain

Shape của bảng này là implementation detail của `langchain_postgres` (đã đổi giữa
các phiên bản: production `public.chat_history` có `session_id text`, còn wrapper
hiện tại tạo `session_id UUID`). DDL viết tay "trông giống" sẽ drift và sinh lỗi
ngụy trang thành bug app. Tạo bằng classmethod của wrapper, qua connection có
`search_path` trỏ sẵn vào schema test:

```python
import os
import psycopg
from langchain_postgres import PostgresChatMessageHistory

conn = psycopg.connect(
    os.environ["TEST_DATABASE_URL"],
    options="-c search_path=regu_test_dsk",
)
PostgresChatMessageHistory.create_tables(conn, "chat_history")  # idempotent
conn.close()
```

`session_id UUID` khác production (`text`): mọi query JOIN giữa `chat_history` và
`conversations` phải cast tường minh (`ch.session_id::text = c.conversation_id`)
để chạy được trên cả hai shape. **Không có teardown drop schema** — schema tạo một
lần, dùng lại giữa các lần chạy; `clean_db` truncate giữa các test là đủ cách ly.

## Cách test pool trỏ vào schema

Mọi pool test (`test_pool`, `async_test_pool`) đặt `search_path` qua **connection
options** (`make_conninfo(..., options="-c search_path=regu_test_dsk")`) chứ không phải
`SET` trong từng test — không code path nào có thể vô tình chạy trên `public`.
Verified qua transaction pooler 6543: `options="-c search_path=..."` sống sót.

## Chạy test suite

```bash
TEST_DATABASE_URL=<url pooler Supabase> uv run pytest tests -v
TEST_DATABASE_URL=<url pooler Supabase> uv run pytest tests/concurrency -m slow -v -s
```
