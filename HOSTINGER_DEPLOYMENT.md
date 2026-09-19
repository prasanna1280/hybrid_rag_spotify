# Hostinger VPS deployment

## 1. Clone the GitHub repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd hybrid_rag_spotify

## 2. Create .env (never commit this file)

nano .env

Set the working OpenAI and Neo4j values from your local setup.

## 3. Build the image

docker compose build

## 4. First-time PDF ingestion

mkdir -p data
docker compose run --rm hybrid-rag python ingest.py --pdf spotify_web_app_architecture.pdf --clear-graph

## 5. Start the application

docker compose up -d

## 6. Verify

docker compose ps
docker compose logs -f hybrid-rag

Test from a browser:
http://YOUR_VPS_IP:8501

For production, put an HTTPS reverse proxy/domain in front of Streamlit rather than exposing 8501 publicly.
