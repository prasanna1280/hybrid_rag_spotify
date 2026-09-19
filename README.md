# Hybrid RAG for the supplied Spotify architecture PDF

This project implements the architecture shown in the supplied diagram:

PDF -> extraction -> section/chunking -> Vector Pipeline + Knowledge Graph
-> Hybrid Retrieval -> Self-RAG answer validation -> Guardrails.

## What is implemented

1. PDF extraction
   - `pypdf`
   - page-aware extraction

2. Section/chunking
   - overlapping chunks
   - page metadata retained

3. Vector pipeline
   - Sentence Transformers embeddings
   - FAISS inner-product similarity
   - normalized embeddings

4. Knowledge Graph
   - Neo4j
   - entities and explicit relationships extracted from chunks
   - chunk-to-entity `MENTIONS` edges
   - entity-to-entity `RELATED_TO` edges

5. Hybrid retrieval
   - vector retrieval
   - graph/entity retrieval
   - reciprocal-rank style fusion

6. Guardrails
   - input length validation
   - prompt-injection pattern checks
   - empty-output checks
   - output size limit
   - source/citation validation

7. Self-RAG
   - generate grounded answer
   - critique grounding
   - revise when unsupported claims are detected

8. Evals
   - retrieval hit-rate baseline
   - editable test cases in `evals/test_cases.json`

## Folder structure

hybrid_rag_spotify/
  app.py
  config.py
  pdf_processor.py
  vector_store.py
  graph_store.py
  llm.py
  guardrails.py
  self_rag.py
  hybrid_rag.py
  ingest.py
  evals.py
  evals/
    test_cases.json
  data/
  .env.example
  requirements.txt

## Setup

### 1. Create environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

Copy `.env.example` to `.env` and provide:

- OPENAI_API_KEY
- NEO4J_URI
- NEO4J_USERNAME
- NEO4J_PASSWORD
- NEO4J_DATABASE

### 4. Ingest the supplied PDF

```bash
python ingest.py --pdf spotify_web_app_architecture.pdf --clear-graph
```

This creates the FAISS index and populates Neo4j.

### 5. Run the application

```bash
streamlit run app.py
```

## Test questions

- Which database stores users, profiles, subscriptions and playlists?
- What service manages playback state?
- What is used for full-text search and autocomplete?
- Which services use PostgreSQL?
- What are the security and compliance capabilities?
- How does the media processor work?
- What is the role of the Recommendation & Personalization Service?

## Important design choice

The vector store is used for semantic similarity, while Neo4j is used to follow
explicit entities and relationships. The final context is fused before generation.

Self-RAG is intentionally applied after retrieval rather than on every low-level
retrieval operation. This keeps the flow understandable:

Input Guardrail
  -> Hybrid Retrieval
  -> Context Assembly
  -> Draft Answer
  -> Self-RAG Critique
  -> Optional Revision
  -> Output Guardrail
  -> Answer

## Production improvements

For a production implementation, consider:

- reranking with a cross-encoder
- entity/relation extraction in batches
- graph schema constraints and indexes
- document/version IDs
- tenant/user-level authorization filters
- prompt-injection detection using a dedicated classifier
- PII/secrets detection
- citation span validation
- answer faithfulness evaluation
- regression test suite in CI/CD
- latency/token/cost tracing
- LangSmith/OpenTelemetry-style observability
- async ingestion workers
- separate vector and graph databases if scale requires it

## Docker deployment

Build:
```bash
docker compose build
```

First-time ingestion:
```bash
mkdir -p data
docker compose run --rm hybrid-rag python ingest.py --pdf spotify_web_app_architecture.pdf --clear-graph
```

Start:
```bash
docker compose up -d
```

Logs:
```bash
docker compose logs -f hybrid-rag
```

The `.env` file is excluded from the Docker build context.
