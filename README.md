# Hybrid RAG — Spotify Architecture Intelligence Assistant

A production-oriented Streamlit application for the supplied Spotify-like web app architecture PDF.

## Architecture

PDF → page-aware extraction/chunking → FAISS vector pipeline + Neo4j knowledge graph → hybrid retrieval → context fusion → LLM → Self-RAG verification → guardrails → answer.

A separate evaluation path uses DeepEval RAG metrics and a configurable 70% pass/target threshold.

## DeepEval

Five RAG metrics are measured:

- Answer Relevancy
- Faithfulness
- Contextual Relevancy
- Contextual Precision
- Contextual Recall

The dashboard also shows deterministic page-hit rate and case pass rate.

Scoring:

- Retrieval Quality = average(Contextual Relevancy, Contextual Precision, Contextual Recall)
- Generation Quality = average(Answer Relevancy, Faithfulness)
- Overall RAG Score = 50% Retrieval + 50% Generation
- Faithfulness is also used as a grounding gate.
- Target threshold = 70% by default.

The score is computed from actual DeepEval results; it is never hard-coded.

## Local setup

```powershell
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
```

Set the OpenAI and Neo4j values in `.env`.

Ingest:

```powershell
python ingest.py --pdf spotify_web_app_architecture.pdf --clear-graph
```

Run:

```powershell
streamlit run app.py
```

Run DeepEval from CLI:

```powershell
python evals.py --deep --cases 10
```

## GitHub → Hostinger → Docker

1. Push the repository to GitHub.
2. On Hostinger VPS, `git pull` the repository.
3. Create `.env` on the VPS; never commit it.
4. Copy the architecture PDF to the project directory.
5. `docker compose build --no-cache`
6. `docker compose run --rm hybrid-rag python ingest.py --pdf spotify_web_app_architecture.pdf --clear-graph`
7. `docker compose up -d`
8. Open `http://YOUR_VPS_IP:8501`.

See `HOSTINGER_DEPLOYMENT.md` for the exact commands.
