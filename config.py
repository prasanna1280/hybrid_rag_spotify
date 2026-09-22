import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = (os.getenv("NEO4J_DATABASE", "neo4j") or "neo4j").strip() or "neo4j"

LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "6"))
TOP_K_GRAPH = int(os.getenv("TOP_K_GRAPH", "6"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "18000"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))
MAX_OUTPUT_CHARS = int(os.getenv("MAX_OUTPUT_CHARS", "8000"))

# DeepEval target. This is a pass threshold, not a fabricated score.
EVAL_THRESHOLD = float(os.getenv("EVAL_THRESHOLD", "0.70"))
EVAL_MAX_CASES = int(os.getenv("EVAL_MAX_CASES", "20"))
DEEPEVAL_MODEL = os.getenv("DEEPEVAL_MODEL", OPENAI_CHAT_MODEL)
