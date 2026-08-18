# backend_migration — Log

## 2026-08-14 — Decouple the core from Streamlit and give FastAPI ownership of the models

Branch `api-migration`, three commits: 17b247c → 9b0ab52 → 33edb1c.

**Where this started.** The backend already owned conversation title generation, and the goal was to move the rest of the chat path (chain invocation, streaming, memory, sidebar SQL) onto FastAPI so that Streamlit becomes UI only. Surveying the code first showed the actual blocker was not on the FastAPI side at all: `src/rag/qa_chain.py`, `src/rag/embedding_utils.py`, `src/rag/reranker_utils.py` and `src/database/connection.py` all imported Streamlit and used `@st.cache_resource`. Nothing could be handed to another process while the build function and the ownership policy were the same decorator.

**What was done, in order.**

*17b247c — literals to config.* The Groq model ids, the per-stage temperatures, the Chroma path and collection name, and the doc-store path moved out of inline literals in `qa_chain.py` into `src/rag/config.py`; `qa_chain` now imports them. Purely mechanical, chosen as the first commit because it is verifiable on its own (Streamlit answers identically) and because it isolates the one file a future model decommissioning has to touch.

*9b0ab52 — de-Streamlit the rag modules.* `@st.cache_resource` and the Streamlit imports came out of the three `src/rag` modules, leaving plain build functions. `get_chain`'s `_reranker_model` parameter was renamed `reranker_model`, since the leading underscore was only ever a Streamlit hashing convention and becomes misleading once the decorator is gone. To keep Streamlit working unchanged, `frontend/services/model_loader.py` was added: a thin `@st.cache_resource` singleton wrapping the now-pure builders, with `frontend/app.py` pointed at it. This had to be one commit — splitting it would leave an intermediate state where Streamlit rebuilds the chain (BM25 construction plus a full `vector_store.get()`) on every rerun.

*33edb1c — models into the FastAPI lifespan.* `lifespan` now builds the embedding model, the reranker and the RAG chain at startup and parks them on `app.state`, and a `get_rag_chain` dependency reads them back, mirroring the existing `get_db_pool` injector. The commit body records two known limitations rather than hiding them: no route consumes the chain yet, and both processes currently hold their own model copies.

**Verification actually run.** All eight changed files parse. `import src.api.main` succeeds against the project venv and reports all seven routes. `inspect.signature(get_chain)` confirms the renamed parameter. The same import also confirmed that `streamlit` is still pulled into the uvicorn process via `src/database/connection.py` — which is why 9b0ab52's message says `src/` is *not yet* fully Streamlit-free.

## Result

`src/rag` is now runtime-agnostic: the same build functions serve Streamlit (through a transitional shim), the FastAPI lifespan, and the existing `evals/` and `ingestion/` scripts. FastAPI owns a built chain on `app.state` and can inject it. Streamlit behaviour is unchanged for the user.

Against the original definition of done — Streamlit sending a request instead of invoking the chain — this is **step zero plus the model-ownership half of step one**. No chat endpoint exists, the memory layer still runs on a Streamlit-cached sync connection, and the frontend still builds and invokes its own chain. What this feature delivered is the precondition that made the rest possible, plus the proof (a uvicorn process that starts and loads) that the decoupling worked.

## 2026-08-15 — Finish de-Streamliting src/ and add the sync memory pool

Branch `api-migration`, three commits: a46a031 → f765c81 → 41a0a58.

**a46a031 — remove Streamlit from `connection.py`.** `src/database/connection.py` is now a plain sync opener (fresh connection per call, no cache, no streamlit import). The process-wide cache + liveness check moved to `frontend/services/connection_provider.py`, a Streamlit-side shim mirroring `model_loader.py`; `app.py` sources its connection factory from there. This closes step 0: importing `src.api.main` no longer pulls `streamlit` into the uvicorn process. Transitional wart, documented in the file's own docstring: the memory path (`get_session_history`) now opens a fresh connection per turn until it moves onto the pool.

**f765c81 — add the sync pool.** New `src/database/sync_connection.py` builds a sync `ConnectionPool` (min 2 / max 10) with `prepare_threshold=None` (Supavisor 6543 guard) and a checkout liveness check. `lifespan` now opens both pools and parks them on `app.state` as `db_pool` (async) and `sync_db_pool` (sync), closing both on shutdown. The async pool got explicit `min_size=2, max_size=5` to replace the hidden default of 4. No consumer yet — the pool exists for the future `POST /chat` handler.

**41a0a58 — delete dead `background_tasks.py`.** Nothing referenced `fire_and_forget`; title generation runs on FastAPI `BackgroundTasks`. Not migrated because `add_script_run_ctx` solves a Streamlit-only session-context problem.

## Result (as of 2026-08-15)

The connection layer now has three deliberate mechanisms: a Streamlit-side cached single connection (`connection.py` + shim), a FastAPI sync pool (`sync_connection.py`, unconsumed), and the FastAPI async pool (title + SQL). `src/` is fully Streamlit-free. The sync pool is built and lifecycle-managed but not yet wired into memory — that is G2/G3. Definition-of-done progress: step 0 complete, the pool infrastructure for the memory path is in place, and the next move is repointing `get_session_history` at the pool and writing the chat endpoint.

## 2026-08-16 — G2: core chain + per-run history binding

Branch `api-migration`, commit 912e5f1. `get_chain` no longer wraps `RunnableWithMessageHistory`; it returns the core chain (route → retrieve → rerank → answer). A new `bind_history(core_chain, conn)` builds the wrapper per run, binding a caller-supplied connection through `partial(get_session_history, conn)`; `get_session_history` was changed to take `(conn, session_id)` and no longer fetches its own connection. This removes the fresh-connection-per-turn wart G0 left in the memory path.

The Streamlit frontend now does the wrap itself: `chat_flow.handle_query` borrows the cached connection from `deps.db_connection_factory()`, calls `bind_history` on the core chain, and passes the wrapped chain into `render_streamed_ai_answer` — `chat_stream.py` is untouched because it still receives an already-wrapped chain and calls `.stream()`. The dead `debug_memory` util was deleted (it called `get_postgres_history` without a connection).

`app.state.rag_chain` now holds the core chain. The G3 handler will apply the same `bind_history` with a pooled connection per request — one helper, two consumers, differing only in connection source and lifecycle. Verified end-to-end in Streamlit: a context-dependent follow-up ("thế còn miễn giảm?") resolves against the prior turn, so read-at-start and write-at-end both work through the new bind path.

## 2026-08-18 — G3: the chat endpoint

Branch `api-migration`, commits c8654f0 → b4f5988 → 600c3f0. The handler predicted at G2 landed: `POST /conversations/{id}/messages` applies the same `bind_history(core_chain, conn)` the frontend uses, but with a connection borrowed from the lifespan sync pool per request — so the pool finally gets its first consumer. The handler is a plain `def`, letting Starlette run the blocking chain in the threadpool rather than on the event loop, and it borrows a connection twice: once briefly to check ownership and eager-create the conversation row, then again to hold across the chain for memory read/write. Two new sync queries back it — `fetch_conversation_owner` (returns owner or None so absent ≠ owned-by-other) and `claim_conversation` (eager-create with title NULL, ON CONFLICT DO NOTHING, so it never clobbers a title the async path later fills). Owner mismatch returns 404 to avoid leaking existence. The response shape is a deliberate projection: `serialize_sources` maps the chain's parent `Document` objects down to `SourceItem` (title/content/doc_id), keeping the wire contract independent of the Document internals.

Verified by hand through Swagger, not curl, using a real uuid4 conversation id: a regulation question returned an answer with non-empty sources; a chitchat question returned empty sources; a same-id follow-up resolved against history (memory across the pooled per-request connection works); a second `X-User-Id` on an existing conversation returned 404; and the title route reported `status: pending, title: null`, confirming the eager-created row exists without a placeholder title. LangSmith traces were correct.

## Result / outcome — G3

The endpoint exists, is committed, and passes the happy path plus the ownership guard. It is **not** proven under concurrency — session isolation, pool starvation, and disconnect behaviour are unmeasured, and that test is the immediate next step (stubbed LLM, shrunk pool, verify via the `chat_history` table). The feature's own definition-of-done — the Streamlit frontend issuing an HTTP request instead of building and invoking its own chain — remains open; G3 is the endpoint that cutover will target.
