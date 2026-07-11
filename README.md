# RAG Pipeline with Hybrid Search

A production-grade Retrieval-Augmented Generation system that ingests internal documentation, indexes it with hybrid search (dense + sparse), retrieves the most relevant context for any question, and generates grounded answers with inline source citations and confidence scoring.

Built from scratch — no LangChain orchestration, no managed RAG platforms. Every component implemented and evaluated independently.

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Answer Correctness | 98.8% |
| IDK Accuracy | 100.0% |
| Citation Accuracy | 97.5% |
| Evaluated On | 50 questions |

*Evaluated on a hand-written golden Q&A dataset covering deployment, rollback, monitoring, incidents, and API documentation.*

---




## Architecture

### Ingestion Flow

Raw Documents (PDF, Markdown, HTML, Text)
↓
Document Loader & Normalizer
↓
Chunking Engine (3 strategies: fixed-size, section-aware, semantic)
↓
Deduplication Layer (cosine similarity > 0.95)
↓
┌─────────────────────┬──────────────────────┐
│  Embedding Model    │   BM25 Index Builder │
│  text-embed-3-small │   rank_bm25          │
│  → 1536-dim vectors │   → inverted index   │
└─────────────────────┴──────────────────────┘
↓                        ↓
ChromaDB                 BM25 pickle
(disk-persisted)         (disk-persisted)



### Query Flow

User Question
↓
Embedding Model (text-embedding-3-small)
↓
┌──────────────────────┬───────────────────────┐
│  Dense Retrieval     │  Sparse Retrieval     │
│  ChromaDB HNSW       │  BM25 keyword scoring │
│  top-10 by cosine    │  top-10 by BM25 score │
└──────────────────────┴───────────────────────┘
↓
Reciprocal Rank Fusion (k=60)
↓
Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
top-20 → top-5
↓
Grounded Generation Prompt
(numbered context blocks, citation instructions)
↓
GPT-4o-mini (temperature=0.0)
↓
Citation Verifier (LLM-as-judge per claim)
↓
Confidence Scorer
(retrieval × 0.35 + citation × 0.40 + completeness × 0.25)
↓
composite ≥ 0.5 → Answer + Citations
composite < 0.5 → "I don't know" response

---



## Tech Stack

| Component | Technology | Why This Choice |
|-----------|------------|-----------------|
| Language | Python 3.11 | Ecosystem standard for ML tooling |
| Embeddings | OpenAI text-embedding-3-small | Best quality/cost ratio, 1536 dimensions |
| Vector Store | ChromaDB 1.5.9 | File-based persistence, HNSW search, no server needed |
| Sparse Search | rank_bm25 | Pure Python BM25, no infrastructure required |
| Rank Fusion | Custom RRF (k=60) | Position-based, no score normalization needed |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Trained on MS MARCO, runs locally on CPU |
| LLM | GPT-4o-mini | 16x cheaper than GPT-4o, sufficient for grounded generation |
| Chunking | LangChain text splitters | Handles edge cases in recursive splitting |
| Data Validation | Pydantic v2 | Type-safe data contracts throughout pipeline |
| API | FastAPI 0.135 | Async-native, automatic OpenAPI docs, Pydantic integration |
| Frontend | Streamlit 1.36 | Rapid UI development in pure Python |
| Containerization | Docker + docker-compose | Reproducible deployment, volume-persisted indexes |

---


## Project Structure
rag-pipeline/
├── backend/
│   ├── ingestion/
│   │   ├── loader.py          # Multi-format document loader (PDF, MD, HTML, TXT)
│   │   ├── chunker.py         # Three chunking strategies
│   │   ├── deduplicator.py    # Cosine similarity deduplication
│   │   └── indexer.py         # Orchestrates ingestion pipeline
│   ├── retrieval/
│   │   ├── dense.py           # ChromaDB vector search
│   │   ├── sparse.py          # BM25 keyword search
│   │   ├── fusion.py          # Reciprocal Rank Fusion
│   │   └── reranker.py        # Cross-encoder reranking
│   ├── generation/
│   │   ├── prompt_builder.py  # Grounded prompt construction
│   │   ├── generator.py       # GPT-4o-mini generation
│   │   ├── citation_parser.py # Extract citations from LLM output
│   │   ├── citation_verifier.py # LLM-as-judge citation verification
│   │   ├── confidence_scorer.py # Composite confidence scoring
│   │   └── pipeline.py        # End-to-end pipeline orchestration
│   ├── evaluation/
│   │   ├── evaluator.py       # Run eval suite against golden dataset
│   │   ├── report.py          # Generate metrics report
│   │   └── strategy_comparison.py # Compare chunking strategies
│   └── api/
│       └── main.py            # FastAPI endpoints
├── frontend/
│   └── app.py                 # Streamlit dashboard
├── data/
│   └── raw/                   # Source documents (drop files here)
├── indexes/
│   ├── chroma/                # ChromaDB vector index (persisted)
│   └── bm25/                  # BM25 index (pickle)
├── evals/
│   └── golden_qa/
│       └── questions.json     # 50-question evaluation dataset
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

---



## Key Design Decisions

### Why Hybrid Search?
Dense retrieval (semantic search) understands meaning but misses exact technical terms — error codes, function names, config keys. Sparse retrieval (BM25) matches exact tokens but misses paraphrased questions. Hybrid search combines both: dense finds intent, BM25 finds specifics. Reciprocal Rank Fusion merges the two ranked lists using position rather than raw scores, which avoids the scale mismatch between cosine similarity and BM25 scores.

### Why a Cross-Encoder Reranker?
The bi-encoder approach used for initial retrieval (embed query and chunk separately, measure distance) is fast but imprecise — it never reads the query and chunk together. The cross-encoder reads both simultaneously, catching nuanced relevance that bi-encoders miss. We apply it only to the top-20 candidates from RRF (not all chunks) to keep latency acceptable.

### Why temperature=0.0?
RAG generation is a grounded reading task, not a creative task. The LLM is given the answer in the context — it just needs to extract and present it accurately. Temperature=0.0 ensures deterministic, consistent responses. The same question always produces the same answer.

### Why LLM-as-judge for citation verification?
Rule-based citation checking (string matching) cannot determine whether a chunk semantically supports a claim. An LLM judge reads both the claim and the cited chunk together and scores the relationship — the same way a human fact-checker would. We use a separate LLM call per citation rather than asking the generating LLM to self-verify (self-verification is unreliable).

### Why separate confidence signals?
A single confidence score would hide where failures occur. By tracking retrieval confidence, citation coverage, and answer completeness separately, we can diagnose failure modes: low retrieval confidence means the corpus lacks relevant content; low citation coverage means the LLM hallucinated; low completeness means the question was only partially answered.

### Chunking Strategy
We evaluated all three chunking strategies (fixed-size, section-aware, semantic) on the 50-question eval suite. Fixed-size and section-aware both achieved 98.8% correctness on our corpus. Semantic chunking was excluded from the comparison due to additional API cost during ingestion. For structured documentation with clear headings, section-aware is preferred in production; for short documents where sections merge together, fixed-size performs equally well.

---



## Notable Bugs Found and Fixed

### 1. ChromaDB Singleton Bug in Deduplicator
**Problem:** When running multiple chunking strategies sequentially in the same Python process, `chromadb.Client()` returned the same in-memory instance across all `Deduplicator` objects. The second strategy's deduplicator checked new chunks against the first strategy's stored chunks, incorrectly flagging valid chunks as duplicates. Section-aware chunking went from 12 chunks to 5 chunks when run after fixed-size.

**Discovery:** Strategy comparison showed section-aware at 53.7% correctness vs fixed-size at 98.8%. After ruling out code differences, traced to chunk count discrepancy (5 vs 12). Isolated the issue by running each strategy alone vs sequentially.

**Fix:** Switched to `chromadb.EphemeralClient()` with unique collection names per instance (`f"dedup_{uuid.uuid4().hex}"`), ensuring true isolation between deduplicator instances.

**Result:** Both strategies correctly produced 12 chunks and achieved identical 98.8% correctness.

### 2. Double Rerank Bug
**Problem:** `full_pipeline()` called `rerank()` to retrieve chunks, then passed the query to `generate()` which called `rerank()` again internally. The cross-encoder model loaded twice and 12 chunks were scored twice per query — doubling retrieval latency.

**Fix:** Made `generate()` accept an optional `chunks` parameter. `full_pipeline()` now calls `rerank()` once and passes chunks directly to all downstream functions.

**Result:** Cross-encoder loads once per query, cutting retrieval time roughly in half.

### 3. Docker Networking — Frontend Cannot Reach API
**Problem:** Frontend container used `API_URL = "http://localhost:8000"`. Inside Docker, `localhost` resolves to the container itself, not the host machine or the API container. Frontend showed "API not reachable."

**Fix:** Made `API_URL` configurable via environment variable. Set `API_URL=http://api:8000` in docker-compose.yml — Docker's internal DNS resolves service names between containers.

**Result:** Frontend correctly routes to API container using Docker's service discovery.

### 4. Hardcoded Document List in Indexer
**Problem:** `run_ingestion()` used a hardcoded `DOCUMENT_FILES` list. Uploading a new document via the API saved it to `data/raw/` but it was never indexed because the list was not updated.

**Fix:** Replaced hardcoded list with dynamic discovery — `DOCUMENTS_DIR.iterdir()` scans `data/raw/` at runtime and indexes all supported file types automatically.

**Result:** Any document uploaded via the API or dropped into `data/raw/` is automatically included in the next ingestion run.

---


## Evaluation

### Golden Dataset
50 questions hand-written against 4 internal documentation files covering deployment procedures, rollback processes, monitoring, incident response, and API reference. Includes 40 answerable questions and 10 unanswerable questions (to test IDK accuracy).

### Metrics

**Answer Correctness (98.8%)**
LLM-as-judge compares system answer against expected answer on a 0-2 scale. Measures whether the key information was correctly retrieved and presented. One question (Q019 — "How do you deploy to production?") scored 1/2 because the system gave a more complete multi-step answer than the narrow expected answer, which the judge penalised. The system's answer was arguably more useful.

**IDK Accuracy (100%)**
All 10 unanswerable questions correctly triggered the "I don't know" path. Confidence scores for unanswerable questions ranged from 0.175 to 0.3 — well below the 0.5 threshold. This metric is critical for production trust: a system that confidently hallucinations wrong answers is more dangerous than one that admits ignorance.

**Citation Accuracy (97.5%)**
Per-claim citation verification using LLM-as-judge. Each claim in the generated answer is checked against its cited chunk to verify the chunk actually supports the claim. 97.5% of citations were fully supported.

### Chunking Strategy Comparison
After fixing the ChromaDB singleton bug, both fixed-size (11 chunks) and section-aware (12 chunks) achieved identical 98.8% correctness on all 50 questions. Our corpus of 4 short documents (~2000 chars each) is too small to differentiate the strategies. On a larger corpus with longer documents, section-aware would be expected to outperform fixed-size by keeping section content together rather than splitting across chunk boundaries.

---


## Setup and Running

### Prerequisites
- Python 3.11+
- Docker and Docker Desktop
- OpenAI API key

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/kreitika/rag-pipeline.git
cd rag-pipeline

# Add your OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env

# Index the sample documents
python -m backend.ingestion.indexer

# Start all services
docker-compose up --build
```

Then open:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

### Local Development (without Docker)

```bash
# Create and activate environment
conda create -n rag-pipeline python=3.11 -y
conda activate rag-pipeline
pip install -r requirements.txt

# Add your OpenAI API key to .env

# Index documents
python -m backend.ingestion.indexer

# Start API (terminal 1)
python -m uvicorn backend.api.main:app --reload --port 8000

# Start frontend (terminal 2)
streamlit run frontend/app.py
```

### Run Evaluation

```bash
python -m backend.evaluation.evaluator
python -m backend.evaluation.report
```

### Add Your Own Documents
Drop any `.md`, `.txt`, `.html`, or `.pdf` file into `data/raw/` and re-run:

```bash
python -m backend.ingestion.indexer
```

Or upload directly through the Streamlit dashboard.

---




## API Reference

### POST /v1/ask
Ask a question against the indexed documentation.

**Request:**
```json
{
  "question": "What is the rollback process if a deployment fails?",
  "top_k": 3
}
```

**Response:**
```json
{
  "question": "What is the rollback process if a deployment fails?",
  "answer": "If a deployment fails...[1]...[2]",
  "confident": true,
  "composite_score": 0.825,
  "citations": [
    {
      "citation_number": 1,
      "source": "deployment_guide.md",
      "chunk_text": "## Rollback Procedures...",
      "times_cited": 1
    }
  ],
  "tokens_used": 653
}
```

### GET /v1/documents
List all indexed documents.

### POST /v1/ingest
Upload a new document for indexing. Accepts multipart/form-data with a `file` field.

### GET /v1/health
Health check endpoint.

---


## Limitations and Future Work

### Current Limitations
- **Small corpus:** Evaluated on 4 sample documents. Performance on large corpora (1000+ documents) is untested but architecturally supported.
- **Sentence splitting:** Semantic chunking uses period-based sentence splitting which struggles with technical content containing version numbers and file paths.
- **Citation parsing:** Claim extraction uses sentence-boundary splitting which misses claims in bullet-point lists.
- **Single-tenant:** No document access control. All users can query all documents.
- **Synchronous ingestion:** Re-indexing blocks the API during ingestion. Production systems would run ingestion asynchronously.

### Potential Improvements
- Add GraphRAG for multi-hop reasoning across related documents
- Implement hierarchical chunking (parent-child) for long documents
- Add embedding caching to avoid re-embedding unchanged chunks
- Build a feedback loop — users mark answers as correct/incorrect to grow the eval dataset
- Add Prometheus metrics and Grafana dashboard for production monitoring
- Implement multi-tenant document isolation with per-user collections
- Fine-tune the embedding model on domain-specific documentation

---

## License
MIT



> Built: July 2026 | Evaluated: 98.8% answer correctness, 100% IDK accuracy, 97.5% citation accuracy on 50 questions