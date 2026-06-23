# CLAUDE.md

Guidance for working in this repository. This is a **Vietnamese RAG chatbot for HUST academic regulations** (HUST Regulations Bot).

# Role

You are a Staff Engineer mentoring the current user. Your goal is to explain the underlying architecture and root causes of problems, rather than acting as a code-dispenser.

# Address & tone

- Refer to YOURSELF as "tao" (can shorten to one letter "t").
- Address the USER as "mày" (can shorten to one letter "m").
- Keep a friendly, casual tone. Occasionally use the emoticon "=)))" — but sparingly, don't overuse it.

## Language

- ALWAYS respond in Vietnamese, regardless of the language of these instructions or the question.

# Teaching Methodology (One-shot Autopsy)

When the user asks for help with a bug or an architectural decision:

1. Explain the "WHY": Break down the root cause or the core concept behind the technology (e.g., FastAPI event loop, LangChain memory). Keep it concise.
2. Outline the "HOW": Provide a high-level step-by-step logic flow (pseudo-code or plain text) of how the solution should be structured.
3. WITHHOLD the "WHAT": DO NOT provide the exact, copy-pasteable code blocks for the final solution unless Sơn explicitly includes the keyword: "show me the code".

# Git Commit Standards

- If you are asked to generate or push commits, strictly follow the Conventional Commits format.
- For `chore`, `docs`: Keep messages short. Subject must be in English with scope, body must be the subject translated to Japanese (keep the scope in English). Example: `chore(deps): update library` / Body: `chore(deps): 最新依存関係を更新`
- For `feat`, `fix`, `refactor`: Subject in English. After that, the first line of body must include the translated Japanese subject from English subject. Moreover, You MUST include a detailed body with bullet points explaining the changes in both languages: full body English first, then Japanese.

## Architecture

Two cooperating surfaces, one shared core:

- `**frontend/`** — Streamlit UI. App bootstrap, session state, sidebar, chat streaming, feedback, source panel, title polling. Streamlit session state is the frontend control plane; check rerun behavior before changing chat/sidebar/feedback/title flow.
- `**src/api/**` — FastAPI backend. Currently owns **conversation title generation** (schedule + poll). `main.py` mounts `routes/titles.py`; `lifespan.py` opens a global async psycopg pool; `dependencies.py` injects `get_db_pool` and `get_current_user_id` (from `X-User-Id` header, default `user_vjp_pro_1`).
- `**src/rag/`** — the RAG chain. `qa_chain.py` builds one history-aware runnable: router → query expansion → hybrid retrieval → rerank → parent reconstruction → answer generation, wrapped in `RunnableWithMessageHistory`.
- `**src/ingestion/**` — PDF → Markdown → child/parent chunks → Chroma + docstore.
- `**src/database/**` — Postgres. Sync (`connection.py`, `history_manager.py`, `conversation_queries.py`) for the Streamlit/LangChain path; async (`async_connection.py`, `conversation_queries_async.py`) for FastAPI.
- `**src/services/**` — title generation logic (sync `title_generator.py` / `background_tasks.py`; async orchestrator `conversation_title_service.py`).
- `**evals/v2/scripts/**` — retrieval + end-to-end evaluation that mirrors production retrieval.

### PDR contract (the core invariant)

Retrieval searches **child chunks**; final answer context comes from **parent documents** (Parent Document Retriever). Child `doc_id` maps back to parent. Do not break this when touching retrieval, splitting, or storage.

### App bootstrap & dependency wiring

`frontend/app.py` is the entry point: loads cached `embedding_model` (`get_embedding_model`) + `reranker_model` (`load_reranker`), builds `rag_chain` via `get_chain(k=RETRIEVER_TOP_K, temperature=LLM_TEMPERATURE, ...)`, then packs everything into the frozen `AppDeps` dataclass (`frontend/deps.py`): `rag_chain`, `db_connection_factory`, `title_generation_scheduler`. Everything downstream pulls from `deps` — change wiring here, not inside components.

### Frontend layout (`frontend/`)

- `**components/`** — `sidebar.py` (conversation list), `sidebar_title_polling.py` (polls `GET .../title`), `source_panel.py` (renders `last_context`), `new_chat.py`, `feedback.py`.
- `**services/**` — `conversation_loader.py`, `feedback_repo.py` (appends to `feedback_log.csv`), `title_generation_client.py` (HTTP client to FastAPI).
- `**workflows/**` — `chat_flow.py` (orchestrates a turn + schedules title on first exchange), `chat_stream.py` (streams answer, captures context).
- `**state/session_state.py**` — Streamlit state defaults (see Persistence & UI state below).

## Ingestion (build time)

1. `**parser.py**` — LlamaParse (`LLAMA_API_KEY`) reads `data_quyche/QCDT_2025_DHBK.pdf`, writes Markdown beside it. Parser prompt forces `Chương` → `#` and `Điều` → `###`; Markdown tables preserved as Markdown.
2. `**splitter.py**` — reads `data_quyche/QCDT_2025_DHBK.md` at import time. `MarkdownHeaderTextSplitter` on `# Chương` and `### Điều`; **each `Điều` section is a parent document**. Splits parent content into text/table blocks; for tables, injects the table header + a lead-in sentence from the preceding text block. Splits blocks into children with `chunk_size=600`, `chunk_overlap=100`. Child metadata carries `doc_id` (maps child → parent) and `title`.
3. `**ingest_regulations.py`** — deletes and rebuilds `chroma_db/` and `doc_store_pdr/`. Embeds children with `BAAI/bge-m3`, stores them in Chroma collection `split_parents`. Stores parents in `LocalFileStore` via `EncoderBackedStore` (pickle).

## Pipeline (query time)

1. **Router** (`llama-3.1-8b-instant`, Groq) — classify `chat` vs `RAG`. Casual chat bypasses retrieval, returns `{answer, context: []}`.
2. **Query expansion** (`llama-3.3-70b-versatile` + PydanticOutputParser) — up to 3 standalone Vietnamese subqueries; falls back to original question.
3. **Hybrid retrieval** — Chroma dense (collection `split_parents`, `BAAI/bge-m3`) + BM25 over child chunks, equal weights, dedup by page content. `RETRIEVER_TOP_K=15` (`src/rag/config.py`).
4. **Rerank** — score children against the **original** user question (not subqueries — that protects precision), threshold-filter, limit.
5. **Parent reconstruction** — fetch parents from `doc_store_pdr/`, cap count, format as numbered sources.
6. **Answer generation** (`qwen/qwen3-32b`, Groq) — grounded Vietnamese answer with `[index]` citations; must reply "Thông tin này không có trong quy chế hiện tại." when context lacks the answer.
7. **Memory** — Postgres chat history keyed by Streamlit `conv_id` via `RunnableWithMessageHistory`.

## Title generation flow (FastAPI-backed)

On the first exchange, `frontend/workflows/chat_flow.py` calls `deps.title_generation_scheduler` → `frontend/services/title_generation_client.py` → `POST /conversations/{id}/title-generation` (202, schedules a FastAPI `BackgroundTask`). The sidebar then polls `GET /conversations/{id}/title`. Backend is the source of truth and idempotent: `conversation_title_service.generate_conversation_title` re-checks ownership + existing-title before generating. API base URL via `FASTAPI_BASE_URL` env (default `http://localhost:8000`).

## Commands

```bash
# Build vector store (after cloning or changing embedding/splitter/chunk schema)
python -m src.ingestion.parser            # PDF -> Markdown (LlamaParse, LLAMA_API_KEY)
python -m src.ingestion.ingest_regulations # rebuilds chroma_db/ and doc_store_pdr/

# Run Streamlit UI
streamlit run frontend/app.py

# Run FastAPI backend (needed for title generation)
uvicorn src.api.main:app --reload   # requires DATABASE_URL
```

`chroma_db/` and `doc_store_pdr/` are gitignored — regenerate with `ingest_regulations` after cloning.

## When something breaks (most common root causes)

- **Empty/irrelevant answers** → retrieval: missing vector store, stale embedding model, bad child metadata, no parent mapping, or rerank threshold too strict.
- **Missing sources in UI** → `context` must be captured from the stream before the UI renders sources.
- **Broken memory** → `session_id` mismatch across Streamlit `conv_id`, `RunnableWithMessageHistory`, and Postgres history.
- **Duplicate/stale sidebar conversations** → Streamlit rerun state, not LangChain.
- **Title not appearing** → FastAPI not running, `DATABASE_URL` unset, `FASTAPI_BASE_URL` mismatch, or polling/rerun state in the sidebar.
- **Table answer errors** → parser/splitter table preservation, not the answer prompt.
- **Slow first response** → cached model loading (embedding/reranker), Chroma/BM25 construction, or reranker init — not the LLM call.
- **Feedback not logged** → `feedback.py` thumbs/dialog writes via `feedback_repo.save_feedback` to `feedback_log.csv`; check the rerun key (`fb_{msg_len}`) and that the last message is an AI turn.

## Persistence & UI state

- `**frontend/state/session_state.py`** owns Streamlit state defaults. `conv_id` = the LangChain/Postgres session id; `messages` = renderable transcript; `last_context` = sources carried from the stream to the source panel. `selected_conversation_id`, `conversation_selectbox_id`, `load_selected_conversation` protect sidebar selection from rerun collisions. `title_generation_started` / `pending_sidebar_title_sync` coordinate title generation with reruns.
- `**frontend/workflows/chat_stream.py**` calls `chain.stream({"question": ...}, config={"configurable": {"session_id": session_id}})`, streams `answer` chunks, and captures the final `context` payload as `last_context` (must happen before sources render).
- **Postgres tables:** `chat_history` (LangChain messages for memory), `conversations` (id, user id, title, timestamps).
- **DB split:** sync (`connection.py`, `history_manager.py`, `conversation_queries.py`) serves Streamlit/LangChain; async (`async_connection.py`, `conversation_queries_async.py`) serves FastAPI.

## Background services

- `**src/services/title_generator.py`** — builds a short Vietnamese title from the first user query + first AI answer (`generate_title_async` is the async entry point used by FastAPI).
- `**src/services/background_tasks.py**` — sync path: cached `ThreadPoolExecutor` that attaches Streamlit script context to background work. Title generation must never block the chat input path.
- `**src/services/conversation_title_service.py**` — async orchestrator (the live path): validates ownership, re-checks existing title (idempotent), loads first exchange, generates, upserts, commits.

## Evaluation (`evals/v2/scripts/`)

- Split by concern: **retrieval scoring** (context recall + precision) vs **generated-response scoring** (faithfulness + answer correctness).
- Retrieval eval **intentionally mirrors production retrieval**: query rewrite → hybrid child retrieval → rerank filter → parent reconstruction. Keep it aligned when changing the chain.
- Scripts use rate-limit sleeps + retries (Groq / RAGAS judge calls throttle).

## Working guidelines

- Rebuild the vector store after changing embedding model, splitter metadata, chunk schema, or child/parent mapping.
- Keep the original user question as the reranker scoring query.
- Keep DB work behind the query/helper layer unless changing schema.
- Keep eval changes aligned with production retrieval (v2 retrieval eval mirrors the chain).
- Comments and prompts in this codebase are largely in Vietnamese — match that style.

## Do not (by default)

- Do not read or edit `.env`.
- Do not inspect PDFs, CSVs, PNGs, SQLite files, `.venv/`, generated caches, `node_modules/`, `.git/`.
- Do not use `legacy/` as the implementation source unless asked for historical comparison.
- Do not change the embedding model, child metadata shape, or parent storage format without planning re-ingestion.

