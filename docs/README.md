# 🤖 HUST Regulations Bot
<p align="center">
  [<a href="./docs/README-JP.md">🇯🇵 日本語</a>]
</p>

[![Watch the video](./assets/DEMO.png)](https://www.youtube.com/watch?v=WOms7keaQgc)
## Overview
To easily query the school's academic regulations (*Quy chế đào tạo*) in natural language, this project aims to help students at **HUST (Hanoi University of Science and Technology)** by a **Retrieval-Augmented Generation (RAG) chatbot**.

The system uses a **Parent-Document-Retriever (PDR)** pattern with **Query Decomposition**, hybrid search (dense vector + BM25), and **Reranking** for high-precision retrieval. A router chain automatically distinguishes between casual conversation and document-grounded Q&A.

---

## Architecture Evolution (v1 to v2)

### Problem of V1 (PDR + ensemble)
The initial version of the RAG system relied on a standard Parent-Document-Retriever (PDR) combined with an Ensemble (Dense + Sparse) search. While functional, it suffered from several critical limitations:

- **High Token Usage:** Token consumption ranged from **2k - 4k tokens** per request. Because the system used a "hard top-k" documents approach without a relevance filter, it frequently passed large, irrelevant parent documents to the LLM, inflating costs and latency.
- **Increasing Noise:** The lack of a reranking stage meant that "hard top-k" retrieval often included "noise documents"-contexts that matched keywords but were irrelevant to the specific legal query.
- **False Positives in Multi-hop Queries:** V1 struggled with queries requiring logical connections between different sections, often getting distracted by keywords without understanding the underlying intent. For example, in corpus dataset:
    - **ID 23:** A freshman asking about conditions for **changing majors** (*chuyển ngành*). V1 retrieved unrelated sections on "dual degrees" and "academic warnings" simply because they shared keywords like "GPA" and "academic conditions."
    - **ID 25:** A query regarding **Engineer graduation ranks** (*hạng tốt nghiệp kỹ sư*) with retaken credits. V1 failed to distinguish between degree levels, retrieving "Master's degree" regulations because of the keyword "graduation rank," missing the specific rules for the Engineer degree.
- **Failed LLM-as-a-Judge Attempts:** Initial attempts to use an LLM to score and filter "fake" or irrelevant contexts proved unsuccessful due to high latency, inconsistent scoring, and the inability to provide a reliable probability-like threshold compared to specialized models.

### V2: Precision & Efficiency (Query Decomposition + PDR + Ensemble + Reranker)
The V2 architecture was redesigned to prioritize precision, speed and context relevance:

- **Multi-stage LLM Orchestration**:
Transitioned from a monolithic LLM approach to a multi-stage specialized model orchestration to balance latency and reasoning quality:

  *  **Intent Routing**: Deployed `llama-3.1-8b-instant` for high-speed semantic classification (Casual Chat vs. RAG).

  * **Query Transformation**: Utilized `llama-3.3-70b-versatile` for complex **query rephrasing** and **decomposition**, enabling effective multi-hop reasoning. This ensures that every intent within a query is addressed by a targeted retrieval pass, significantly improving recall for nuanced questions.

  * **Grounded Inference**: Employed `qwen/qwen3-32b` as the core engine for generating accurate, citation-backed answers.

- **Cross-Encoder Reranker:** Integrated the `BAAI/bge-reranker-v2-m3` model to score the relevance of all retrieved child chunks against the original question. 
- **Noise Cancellation & Token Optimization:** By applying a **0.5 relevance threshold** on reranker scores, the reranker model effectively "cancel" semantic gap and false positive before they reach the LLM. This has **reduced token usage by 50%**, bringing the average down to **800 - 2k tokens** while improving answer quality.
- **Complex Table Preservation:** Leveraging LlamaParse, V2 preserves the complex format of **Markdown tables**. This is crucial for university regulations where critical data - such as credit limits, grade conversion formulas, and graduation criteria - is often stored in tabular form.

### Evaluation Metrics
The transition from V1 to V2 shows a clear trade-off between recall and precision, with V2 drastically reducing noise.

| Metric | V1 (Baseline) | V2 (Optimized) | Impact |
| :--- | :--- | :--- | :--- |
| **Context Recall** | 0.98 | 0.94 | Slight decrease due to stricter filtering thresholds. |
| **Context Precision** | 0.84 | **0.9756** | **+13.5%** improvement; validates the Reranker's ability to eliminate noise. |
| **Faithfulness** | Baseline focus on retrieval | 0.7753 | High level of adherence to the retrieved context. |
| **Answer Correctness**| Baseline focus on retrieval | 0.6315 | Solid baseline for end-to-end response accuracy. |

## How It Works
### V1: Baseline Architecture
![V1 Architecture Diagram](assets/DIAGRAM.png)

### V2: Advanced Pipeline
![V1 Architecture Diagram](assets/DIAGRAM_V2.png)

```
User question
      │
      ▼
 Router Chain ──► "chat"  ──► Friendly chat response 
                                          │
                                          ▼ 
                             Memory + User Feedback
      │
      └──────► "RAG"
                  │
                  ▼
         Query Decomposition [V2: Decompose into standalone sub-queries]
                  │
                  ▼
         EnsembleRetriever (Dense + Sparse)
         ├── Dense: ChromaDB vector search  (weight 0.5)
         └── Sparse: BM25 keyword search    (weight 0.5)
                  │
                  ▼
         Cross-Encoder Reranker [V2: Bridging Semantic Gap & Filtering Noise]
                  │
                  ▼
         Top-k filtered doc_ids → fetch full parent docs (PDR Pattern)
                  │
                  ▼
         QA chain (citation-enforced) → streaming answer + sources
                  │
                  ▼
         Store into memory conversation + User feedback about response
```


## Prerequisites

| Requirement | Version / Notes |
|---|---|
| Python | `3.13+` (specific ver: `3.13.5`) |
| Groq API Key | Required - powers the LLM (`qwen/qwen3-32b`) |
| LlamaParse API Key| Required - for high-accuracy PDF document parsing |
| LangSmith API Key | Optional - for chain tracing |
| GPU (CUDA) | Optional - auto-detected (2GB VRAM minimum usage) ; CPU is the fallback for embeddings |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/mizuwonomu/rag-project.git
cd rag-project
```

### 2. Set up environment variables
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key
LLAMA_API_KEY=your_llama_api_key
LANGSMITH_API_KEY=your_langsmith_api_key   # Optional
```

### 3. Install dependencies
`uv` is preferred:
```bash
uv sync
# or with pip:
pip install -r requirements.txt
```

### 4. Build the vector store
* Create a `data_quyche/` folder inside project root.
* Place the source PDF [QCDT_2025_DHBK.pdf](https://ctt.hust.edu.vn/Upload/Nguy%E1%BB%85n%20Qu%E1%BB%91c%20%C4%90%E1%BA%A1t/files/DTDH_QDQC/Hoctap/QCDT_2025_5445_QD-DHBK.pdf) inside `data_quyche/`.
* Run the parser to convert PDF to structured Markdown:
```bash
python -m src.ingestion.parser
```
This will create a structured data markdown file inside the `data_quyche/` folder.
* After that, run the ingestion script to build the vector store:
```bash
python -m src.ingestion.ingest_regulations
```
This will:
- Split the markdown file into blocks with distinct table, text blocks.
- For tables, it will inject metadata with lead-in sentence, preserve header for each table.
- For texts, it will inject metadata of "Chuong"- "Dieu" for each child chunks to preserve parent's context.
- Create *child* chunks (`chunk_size=600`, `chunk_overlap=100`) and embed them into ChromaDB (`chroma_db/` folder)
- Persist parent docs as **pickled bytes** in `doc_store_pdr/` (This will be generated automatically)

### 5. Run the app
```bash
streamlit run frontend/app.py
```

---


## Project Structure

```
rag-project/
├── evals/ # Evaluation datasets, scripts, results, metrics
├── frontend/
│   └── app.py                    # Streamlit frontend entry-point
├── src/
│   ├── __init__.py
│   ├── qa_chain.py               # Core RAG/chat chain logic
    ├── reranker_utils.py # Shared utilities (reranker model loader)
│   ├── utils.py                  # Shared utilities (embedding model loader)
│   └── ingestion/
│       ├── __init__.py
│       └── ingest_regulations.py # PDF ingestion & vector store population
        └── parser.py # LlamaParse PDF parser
        └── splitter.py # Markdown splitter with metadata injection
├── data_quyche/                  # Raw PDF documents (not versioned)
├── chroma_db/                    # ChromaDB vector store (auto-generated, not versioned)
├── doc_store_pdr/                # Parent doc store (auto-generated, not versioned)
├── assets/                       # Static assets
├── docs/                         # Additional documentation
├── legacy/                       # Archived code — do not use
├── feedback_log.csv              # User feedback log (auto-created, not versioned)
├── .env                          # API keys — never commit
├── pyproject.toml                # Project metadata & pinned dependencies
├── requirements.txt              # pip install manifest
└── uv.lock                       # uv lockfile — do not hand-edit
```

---

## Key Design Decisions

| Component | Choice | Reason |
|---|---|---|
| Embedding model | `BAAI/bge-m3` (HuggingFace) | Strong multilingual (Vietnamese) performance |
| Vector store | ChromaDB | Persistent, local, no server needed |
| Parent store | `LocalFileStore` + pickle | Preserves full article context per "Điều" |
| Retrieval | EnsembleRetriever (dense + BM25) | Hybrid search balances semantic & keyword recall |
| Retrieval | `BAAI/bge-reranker-v2-m3` | Cross-encoder Reranker - high docs ranking performance  |
| LLM | `ChatGroq` - `qwen/qwen3-32b` | Fast inference via Groq API |
| Memory | In-process `RunnableWithMessageHistory` | Lightweight single-session support |

---

## Notes

- `chroma_db/` and `doc_store_pdr/` are excluded from version control. Regenerate them with `python -m src.ingestion.ingest_regulations` after cloning.
- Never change the embedding model without re-running ingestion - the vector store and model must match.
- Keep `.env` file secure and never commit it.
