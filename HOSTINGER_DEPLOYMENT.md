# Hostinger VPS deployment — Hybrid RAG v3

## 1. Clone the GitHub repository

```bash
cd /opt
git clone <YOUR_GITHUB_REPO_URL> hybrid_rag_spotify
cd /opt/hybrid_rag_spotify
```

For an existing clone:

```bash
cd /opt/hybrid_rag_spotify
git pull origin main
```

## 2. Add environment variables

Create the file locally on the VPS. Do not commit it:

```bash
nano .env
```

Use the values from `.env.example`. Keep secrets only in `.env`.

## 3. Put the source PDF on the VPS

The application expects the supplied PDF for ingestion. Copy it into the project directory:

```bash
ls -lh spotify_web_app_architecture.pdf
```

## 4. Build the Docker image

```bash
docker compose build --no-cache
```

## 5. Ingest the PDF into FAISS + Neo4j

Run this once after a clean/new ingestion:

```bash
docker compose run --rm hybrid-rag python ingest.py --pdf spotify_web_app_architecture.pdf --clear-graph
```

## 6. Start the application

```bash
docker compose up -d
```

## 7. Verify

```bash
docker compose ps
docker compose logs -f hybrid-rag
```

Open:

`http://YOUR_VPS_IP:8501`

## 8. Run DeepEval from the Streamlit UI

Open **Deep Evaluation** → choose 5–24 cases → **Run DeepEval**.

The report is saved to:

`data/eval_latest.json`

The 70% setting is controlled by:

`EVAL_THRESHOLD=0.70`

It is a real pass/target threshold. The application does not fabricate scores above 70%.

## 9. CLI evaluation option

```bash
docker compose run --rm hybrid-rag python evals.py --deep --cases 10
```

## 10. Updating the application

After pushing changes to GitHub:

```bash
cd /opt/hybrid_rag_spotify
git pull origin main
docker compose build --no-cache
docker compose up -d
```

If ingestion/chunking/graph extraction changed, rebuild the indexes:

```bash
docker compose run --rm hybrid-rag python ingest.py --pdf spotify_web_app_architecture.pdf --clear-graph
docker compose restart hybrid-rag
```

## 11. Important security notes

- Never commit `.env`.
- Never put OpenAI or Neo4j passwords in source files.
- Restrict port 8501 with your VPS firewall if the application should not be public.
- For production HTTPS, put Nginx/Caddy/Traefik in front of Streamlit.
