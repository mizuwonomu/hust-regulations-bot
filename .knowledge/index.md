# Knowledge Index — HUST Regulations Bot

Vietnamese RAG chatbot over HUST academic regulations (Quy chế đào tạo).

Cross-cutting pointers (read before touching the named layer):
- **Reranker score is NOT comparable across queries** — a cross-encoder score of 0.9 and 0.0007 can both mean "most relevant doc for this query". Any logic that treats an absolute reranker score as a global threshold is unsound. This killed the hard-floor design and constrains every future gating idea. See `features/rerank_ratio/decisions.md`.
- **Post-rerank child selection is a relative ratio to top-1** (`RERANK_RATIO = 0.6`, `src/rag/config.py`), applied in `retreive_parents._apply_score_ratio` (commit 9f48018, 2026-08-07). There is no absolute floor and no out-of-scope gate in the rerank layer; empty context now means only "candidate pool was empty". See `features/rerank_ratio/`.
- **PDR contract** — retrieval searches *child* chunks; final answer context is *parent* documents. Child `doc_id` maps back to parent. Breaking this breaks retrieval, splitting and storage at once.

## Architecture (current)

Two cooperating surfaces over one shared core.

**`frontend/` — Streamlit UI.** The entry point `frontend/app.py` loads the cached embedding model (`get_embedding_model`) and reranker (`load_reranker`), builds the chain via `get_chain(k=RETRIEVER_TOP_K, temperature=LLM_TEMPERATURE, ...)`, and packs everything into the frozen `AppDeps` dataclass (`frontend/deps.py`): `rag_chain`, `db_connection_factory`, `title_generation_scheduler`. Everything downstream pulls from `deps` — wiring changes belong here, not inside components. Components: `sidebar.py` (conversation list), `sidebar_title_polling.py` (polls `GET .../title`), `source_panel.py` (renders `last_context`), `new_chat.py`, `feedback.py`. Services: `conversation_loader.py`, `feedback_repo.py` (appends to `feedback_log.csv`), `title_generation_client.py`. Workflows: `chat_flow.py` (orchestrates a turn, schedules title on first exchange) and `chat_stream.py` (streams the answer, captures context). Streamlit session state is the frontend control plane — check rerun behaviour before changing chat/sidebar/feedback/title flow.

**`src/api/` — FastAPI backend.** Currently owns conversation title generation only. `main.py` mounts `routes/titles.py`; `lifespan.py` opens a global async psycopg pool; `dependencies.py` injects `get_db_pool` and `get_current_user_id` (from the `X-User-Id` header, default `user_vjp_pro_1`).

**`src/rag/` — the chain.** `qa_chain.py` builds one history-aware runnable: router → query expansion → hybrid retrieval → rerank → parent reconstruction → answer generation, wrapped in `RunnableWithMessageHistory`. Query-time pipeline: (1) router (`llama-3.1-8b-instant`, Groq) classifies `chat` vs `RAG`, casual chat bypasses retrieval and returns `{answer, context: []}`; (2) query expansion (`llama-3.3-70b-versatile` + `PydanticOutputParser`) produces up to 3 standalone Vietnamese subqueries, falling back to the original question; (3) hybrid retrieval — Chroma dense (collection `split_parents`, `BAAI/bge-m3`) plus BM25 over child chunks, equal weights, dedup by page content, `RETRIEVER_TOP_K=15`; (4) rerank — every deduped candidate is scored against the **original** user question (not the subqueries — subqueries buy recall, the reranker must protect precision), then selected by the relative ratio rule above and capped by `RERANK_MAX_CHILDREN`; (5) parent reconstruction — parents fetched from `doc_store_pdr/`, capped at 4, formatted as numbered sources; (6) answer generation (Groq) produces a grounded Vietnamese answer with `[index]` citations and must reply "Thông tin này không có trong quy chế hiện tại." when context lacks the answer; (7) memory — Postgres chat history keyed by the Streamlit `conv_id` via `RunnableWithMessageHistory`. Rerank scoring and selection are exposed as LangSmith spans (`rerank_children`, `apply_score_ratio`, `fetch_parents`).

**`src/ingestion/` — build time.** `parser.py` runs LlamaParse (`LLAMA_API_KEY`) over `data_quyche/QCDT_2025_DHBK.pdf` and writes Markdown beside it; the parser prompt forces `Chương` → `#` and `Điều` → `###`, and Markdown tables stay Markdown. `splitter.py` reads the Markdown at import time and uses `MarkdownHeaderTextSplitter` on `# Chương` and `### Điều` — **each `Điều` section is a parent document**. Parent content is split into text/table blocks; for tables it injects the table header plus a lead-in sentence from the preceding text block. Blocks become children at `chunk_size=600`, `chunk_overlap=100`, with `doc_id` (child → parent) and `title` in child metadata. `ingest_regulations.py` deletes and rebuilds `chroma_db/` and `doc_store_pdr/`: children embedded with `BAAI/bge-m3` into Chroma collection `split_parents`, parents pickled into a `LocalFileStore` via `EncoderBackedStore`. Both stores are gitignored — regenerate after cloning.

**`src/database/` — Postgres, deliberately split.** Sync (`connection.py`, `history_manager.py`, `conversation_queries.py`) serves the Streamlit/LangChain path; async (`async_connection.py`, `conversation_queries_async.py`) serves FastAPI. Tables: `chat_history` (LangChain messages for memory) and `conversations` (id, user id, title, timestamps). Keep DB work behind the query/helper layer unless changing schema.

**`src/services/` — background work.** `title_generator.py` builds a short Vietnamese title from the first user query + first AI answer (`generate_title_async` is the async entry point). `background_tasks.py` is the sync path: a cached `ThreadPoolExecutor` that attaches the Streamlit script context to background work — title generation must never block the chat input path. `conversation_title_service.py` is the live async orchestrator: validates ownership, re-checks for an existing title (idempotent), loads the first exchange, generates, upserts, commits.

**Title generation flow.** On the first exchange `chat_flow.py` calls `deps.title_generation_scheduler` → `title_generation_client.py` → `POST /conversations/{id}/title-generation` (202, schedules a FastAPI `BackgroundTask`). The sidebar then polls `GET /conversations/{id}/title`. The backend is the source of truth and idempotent. Base URL via `FASTAPI_BASE_URL` (default `http://localhost:8000`).

**`evals/v2/scripts/` — evaluation.** Split by concern: retrieval scoring (context recall + precision) vs generated-response scoring (faithfulness + answer correctness). The retrieval eval **intentionally mirrors production retrieval** (query rewrite → hybrid child retrieval → rerank selection → parent reconstruction) and must be kept aligned when the chain changes. Scripts use rate-limit sleeps and retries (Groq / RAGAS judge calls throttle). Rerank calibration lives in `run_rewrite_rerank_calibration.py`, which has a single-distribution mode and an A/B mode (formal vs abbreviated phrasing, paired by id).

## Persistence & UI state

`frontend/state/session_state.py` owns Streamlit state defaults. `conv_id` is the LangChain/Postgres session id; `messages` is the renderable transcript; `last_context` carries sources from the stream to the source panel. `selected_conversation_id`, `conversation_selectbox_id` and `load_selected_conversation` protect sidebar selection from rerun collisions; `title_generation_started` / `pending_sidebar_title_sync` coordinate title generation with reruns. `chat_stream.py` calls `chain.stream({"question": ...}, config={"configurable": {"session_id": session_id}})`, streams `answer` chunks, and captures the final `context` payload as `last_context` — this must happen before sources render.

## Commands

```bash
python -m src.ingestion.parser             # PDF -> Markdown (LlamaParse, LLAMA_API_KEY)
python -m src.ingestion.ingest_regulations # rebuilds chroma_db/ and doc_store_pdr/
streamlit run frontend/app.py              # Streamlit UI
uvicorn src.api.main:app --reload          # FastAPI backend (requires DATABASE_URL)
```

## When something breaks (most common root causes)

- **Empty/irrelevant answers** → retrieval: missing vector store, stale embedding model, bad child metadata, no parent mapping, or the rerank ratio too strict.
- **Missing sources in UI** → `context` must be captured from the stream before the UI renders sources.
- **Broken memory** → `session_id` mismatch across Streamlit `conv_id`, `RunnableWithMessageHistory`, and Postgres history.
- **Duplicate/stale sidebar conversations** → Streamlit rerun state, not LangChain.
- **Title not appearing** → FastAPI not running, `DATABASE_URL` unset, `FASTAPI_BASE_URL` mismatch, or polling/rerun state in the sidebar.
- **Table answer errors** → parser/splitter table preservation, not the answer prompt.
- **Slow first response** → cached model loading (embedding/reranker), Chroma/BM25 construction, or reranker init — not the LLM call.
- **Feedback not logged** → `feedback.py` writes via `feedback_repo.save_feedback` to `feedback_log.csv`; check the rerun key (`fb_{msg_len}`) and that the last message is an AI turn.

## Working guidelines

- Rebuild the vector store after changing embedding model, splitter metadata, chunk schema, or child/parent mapping.
- Keep the original user question as the reranker scoring query.
- Keep DB work behind the query/helper layer unless changing schema.
- Keep eval changes aligned with production retrieval.
- Comments and prompts in this codebase are largely in Vietnamese — match that style.

## Do not (by default)

- Do not read or edit `.env`.
- Do not inspect PDFs, CSVs, PNGs, SQLite files, `.venv/`, generated caches, `node_modules/`, `.git/`.
- Do not use `legacy/` as the implementation source unless asked for historical comparison.
- Do not change the embedding model, child metadata shape, or parent storage format without planning re-ingestion.
