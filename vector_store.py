import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import LOCAL_EMBEDDING_MODEL


class VectorStore:
    def __init__(self, index_path="data/faiss.index", metadata_path="data/chunks.json"):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        self.index = None
        self.metadata = []

        if self.index_path.exists() and self.metadata_path.exists():
            self.load()

    def build(self, chunks):
        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.metadata = chunks
        self.save()

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(
            self.metadata_path.read_text(encoding="utf-8")
        )

    def search(self, query, k=5):
        if self.index is None:
            raise RuntimeError("FAISS index is not built. Run ingest.py first.")

        q = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        scores, ids = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)

        return results
