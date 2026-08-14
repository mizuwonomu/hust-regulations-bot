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
