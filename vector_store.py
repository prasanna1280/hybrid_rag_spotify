import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import LOCAL_EMBEDDING_MODEL


class VectorStore:
    """
    Hybrid vector store supporting:

    1. FAISS semantic search
    2. BM25 keyword / lexical search
    3. Architecture-aware query expansion
    4. Architecture service-name boosting
    """

    def __init__(
        self,
        index_path="data/faiss.index",
        metadata_path="data/chunks.json",
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

        self.index = None
        self.metadata = []

        # BM25 state
        self.bm25 = None
        self.bm25_corpus = []

        if self.index_path.exists() and self.metadata_path.exists():
            self.load()

    # ============================================================
    # TOKENIZATION
    # ============================================================

    @staticmethod
    def _tokenize(text):
        """
        Tokenize text for BM25.

        Keeps useful technical tokens such as:
        - full-text
        - Elasticsearch
        - OpenSearch
        - PostgreSQL
        - HLS
        - API
        """
        text = text.lower()

        tokens = re.findall(
            r"[a-z0-9]+(?:[-_/][a-z0-9]+)*",
            text,
        )

        return tokens

    # ============================================================
    # BUILD VECTOR INDEX
    # ============================================================

    def build(self, chunks):
        """
        Build the FAISS semantic index and BM25 keyword index.
        """

        texts = [c["text"] for c in chunks]

        # --------------------------------------------------------
        # FAISS semantic embeddings
        # --------------------------------------------------------

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        self.metadata = chunks

        # --------------------------------------------------------
        # BM25 keyword index
        # --------------------------------------------------------

        self._build_bm25()

        # --------------------------------------------------------
        # Persist indexes
        # --------------------------------------------------------

        self.save()

    # ============================================================
    # BM25 INDEX
    # ============================================================

    def _build_bm25(self):
        """
        Build BM25 lexical search index from chunk text.
        """

        if not self.metadata:
            self.bm25 = None
            self.bm25_corpus = []
            return

        self.bm25_corpus = [
            self._tokenize(item.get("text", ""))
            for item in self.metadata
        ]

        self.bm25 = BM25Okapi(self.bm25_corpus)

    # ============================================================
    # SAVE
    # ============================================================

    def save(self):
        """
        Save FAISS index and chunk metadata.
        """

        if self.index is None:
            raise RuntimeError("Cannot save empty FAISS index.")

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        self.metadata_path.write_text(
            json.dumps(
                self.metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ============================================================
    # LOAD
    # ============================================================

    def load(self):
        """
        Load FAISS index and metadata.

        BM25 is rebuilt from the stored metadata.
        """

        self.index = faiss.read_index(
            str(self.index_path)
        )

        self.metadata = json.loads(
            self.metadata_path.read_text(
                encoding="utf-8"
            )
        )

        # Rebuild BM25 whenever the application starts.
        self._build_bm25()

    # ============================================================
    # SEMANTIC SEARCH
    # ============================================================

    def semantic_search(self, query, k=5):
        """
        Semantic similarity search using FAISS.
        """

        if self.index is None:
            raise RuntimeError(
                "FAISS index is not built. "
                "Run ingest.py first."
            )

        q = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        scores, ids = self.index.search(
            q,
            k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            ids[0],
        ):
            if idx < 0:
                continue

            item = dict(self.metadata[idx])

            item["score"] = float(score)

            # Explicit semantic score for hybrid ranking.
            item["semantic_score"] = float(score)

            results.append(item)

        return results

    # ============================================================
    # BACKWARD-COMPATIBLE SEARCH
    # ============================================================

    def search(self, query, k=5):
        """
        Backward-compatible semantic search.

        Existing code using:
            vector.search(...)

        will continue to work.
        """

        return self.semantic_search(
            query,
            k,
        )

    # ============================================================
    # KEYWORD SEARCH - BM25
    # ============================================================

    def keyword_search(self, query, k=5):
        """
        BM25 keyword / lexical search with:

        - Architecture-aware query expansion
        - Exact technology boosting
        - Architectural service-name boosting
        """

        if self.bm25 is None:
            self._build_bm25()

        if self.bm25 is None:
            return []

        query_lower = query.lower()

        query_tokens = self._tokenize(query)

        # --------------------------------------------------------
        # Architecture-aware query expansion
        # --------------------------------------------------------

        if (
            "autocomplete" in query_lower
            or "full-text" in query_lower
            or "full text" in query_lower
        ):
            query_tokens += [
                "elasticsearch",
                "opensearch",
                "search",
                "autocomplete",
            ]

        if (
            "recommendation" in query_lower
            or "recommendations" in query_lower
        ):
            query_tokens += [
                "recommendation",
                "recommendations",
                "embedding",
                "embeddings",
                "listening",
            ]

        if "playback" in query_lower:
            query_tokens += [
                "playback",
                "audio",
                "session",
                "sessions",
            ]

        if (
            "database" in query_lower
            or "databases" in query_lower
        ):
            query_tokens += [
                "postgresql",
                "redis",
                "database",
            ]

        if "streaming" in query_lower:
            query_tokens += [
                "streaming",
                "hls",
                "media",
                "delivery",
            ]

        if (
            "media processor" in query_lower
            or "transcoding" in query_lower
        ):
            query_tokens += [
                "media",
                "processor",
                "ffmpeg",
                "transcoding",
                "hls",
            ]

        if (
            "object storage" in query_lower
            or "storage" in query_lower
        ):
            query_tokens += [
                "s3",
                "object",
                "storage",
                "hls",
            ]

        # --------------------------------------------------------
        # BM25 scoring
        # --------------------------------------------------------

        scores = self.bm25.get_scores(
            query_tokens
        )

        # --------------------------------------------------------
        # Technology / architecture exact-term boost
        # --------------------------------------------------------

        boost_terms = {
            "postgresql",
            "redis",
            "elasticsearch",
            "opensearch",
            "neo4j",
            "clickhouse",
            "kafka",
            "nats",
            "rabbitmq",
            "graphql",
            "websocket",
            "ffmpeg",
            "next.js",
            "react",
            "typescript",
            "tailwind",
            "docker",
            "kubernetes",
            "prometheus",
            "grafana",
            "opentelemetry",
            "hls",
        }

        query_boost_terms = {
            token
            for token in query_tokens
            if token in boost_terms
        }

        for idx in range(len(scores)):
            text_lower = self.metadata[idx].get(
                "text",
                "",
            ).lower()

            for term in query_boost_terms:

                # Handle slash-separated technologies.
                if term in text_lower:
                    scores[idx] += 2.0

        # --------------------------------------------------------
        # Architectural service-name boost
        #
        # Example:
        # "What relationships does the Catalog Service manage?"
        #
        # -> catalog service
        #
        # "How does the Recommendation & Personalization
        # Service generate recommendations?"
        #
        # -> recommendation personalization service
        # --------------------------------------------------------

        service_candidates = self._extract_service_candidates(
            query_lower
        )

        if service_candidates:

            for idx in range(len(scores)):

                text_lower = self.metadata[idx].get(
                    "text",
                    "",
                ).lower()

                for candidate in service_candidates:

                    if candidate in text_lower:
                        scores[idx] += 4.0

        # --------------------------------------------------------
        # Final ranking
        # --------------------------------------------------------

        ranked = []

        for idx, score in enumerate(scores):

            if score > 0:

                ranked.append(
                    (
                        idx,
                        float(score),
                    )
                )

        ranked.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        # --------------------------------------------------------
        # Return metadata + keyword score
        # --------------------------------------------------------

        results = []

        for idx, score in ranked[:k]:

            item = dict(
                self.metadata[idx]
            )

            item["keyword_score"] = score

            results.append(item)

        return results

    # ============================================================
    # SERVICE NAME EXTRACTION
    # ============================================================

    @staticmethod
    def _extract_service_candidates(query):
        """
        Extract likely architectural service-name phrases.

        This intentionally avoids treating generic phrases such as:

            "Which service..."
            "What service..."

        as service names.

        Examples:

            "What relationships does the Catalog Service manage?"
                -> ["catalog service"]

            "How does the Recommendation & Personalization
             Service generate recommendations?"
                -> ["recommendation personalization service"]

            "What does the Media Processor Service do?"
                -> ["media processor service"]
        """

        if "service" not in query:
            return []

        # Common words that should not form part of a service name.
        stop_words = {
            "which",
            "what",
            "how",
            "does",
            "do",
            "is",
            "are",
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "for",
            "to",
            "in",
            "on",
            "with",
            "from",
            "used",
            "use",
            "stores",
            "store",
            "manage",
            "manages",
            "handle",
            "handles",
            "generate",
            "generates",
            "support",
            "supports",
            "listed",
            "role",
            "responsibilities",
            "technology",
            "technologies",
        }

        candidates = []

        # --------------------------------------------------------
        # Find every occurrence of "service".
        # --------------------------------------------------------

        service_positions = [
            m.start()
            for m in re.finditer(
                r"\bservice\b",
                query,
            )
        ]

        for position in service_positions:

            prefix = query[:position].strip()

            tokens = re.findall(
                r"[a-z0-9]+",
                prefix,
            )

            # Remove generic question words.
            meaningful = [
                token
                for token in tokens
                if token not in stop_words
            ]

            # Keep only the last few terms before "service".
            meaningful = meaningful[-4:]

            if not meaningful:
                continue

            candidate = (
                " ".join(meaningful)
                + " service"
            )

            candidates.append(candidate)

        # --------------------------------------------------------
        # Also support common slash/& notation.
        # --------------------------------------------------------

        normalized_query = (
            query
            .replace("&", " ")
            .replace("/", " ")
            .replace("-", " ")
        )

        service_positions = [
            m.start()
            for m in re.finditer(
                r"\bservice\b",
                normalized_query,
            )
        ]

        for position in service_positions:

            prefix = normalized_query[
                :position
            ].strip()

            tokens = re.findall(
                r"[a-z0-9]+",
                prefix,
            )

            meaningful = [
                token
                for token in tokens
                if token not in stop_words
            ]

            meaningful = meaningful[-4:]

            if not meaningful:
                continue

            candidate = (
                " ".join(meaningful)
                + " service"
            )

            candidates.append(candidate)

        # --------------------------------------------------------
        # Remove duplicates
        # --------------------------------------------------------

        unique_candidates = []

        for candidate in candidates:

            if candidate not in unique_candidates:
                unique_candidates.append(
                    candidate
                )

        return unique_candidates