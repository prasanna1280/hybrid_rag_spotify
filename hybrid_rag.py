from config import TOP_K_VECTOR, TOP_K_GRAPH, MAX_CONTEXT_CHARS
from guardrails import validate_input, validate_output, enforce_citations
from graph_store import KnowledgeGraph
from llm import json_chat
from self_rag import generate_answer, critique, revise
from vector_store import VectorStore


ENTITY_SYSTEM = """
Extract important entities from the user question for knowledge-graph retrieval.
Return JSON:
{"entities": [{"name": "...", "type": "..."}]}
Keep only entities that are actually present in the question.
"""


class HybridRAG:
    def __init__(self):
        self.vector = VectorStore()
        self.graph = KnowledgeGraph()

    def extract_entities(self, question):
        result = json_chat(ENTITY_SYSTEM, question)
        return result.get("entities", [])

    def retrieve(self, question):
        vector_results = self.vector.search(question, TOP_K_VECTOR)

        entities = self.extract_entities(question)
        entity_names = [e["name"] for e in entities]

        graph_results = self.graph.search_entities(
            entity_names,
            limit=TOP_K_GRAPH,
        )

        # Reciprocal-rank style fusion.
        merged = {}

        for rank, item in enumerate(vector_results, start=1):
            key = item["chunk_id"]
            merged.setdefault(
                key,
                {
                    "chunk_id": key,
                    "page": item["page"],
                    "text": item["text"],
                    "vector_score": 0,
                    "graph_score": 0,
                    "graph_hits": [],
                },
            )
            merged[key]["vector_score"] += 1 / (60 + rank)

        for rank, item in enumerate(graph_results, start=1):
            key = item["chunk_id"]
            merged.setdefault(
                key,
                {
                    "chunk_id": key,
                    "page": item["page"],
                    "text": item["text"],
                    "vector_score": 0,
                    "graph_score": 0,
                    "graph_hits": [],
                },
            )
            merged[key]["graph_score"] += 1 / (60 + rank)
            merged[key]["graph_hits"].append(item)

        ranked = sorted(
            merged.values(),
            key=lambda x: x["vector_score"] + x["graph_score"],
            reverse=True,
        )

        return ranked

    def build_context(self, results):
        blocks = []
        total = 0

        for i, item in enumerate(results, start=1):
            block = (
                f"[Source {i} | Page {item['page']} | Chunk {item['chunk_id']}]\n"
                f"{item['text']}\n"
            )

            if total + len(block) > MAX_CONTEXT_CHARS:
                break

            blocks.append(block)
            total += len(block)

        return "\n".join(blocks)

    def ask(self, question):
        ok, checked = validate_input(question)
        if not ok:
            return {"answer": checked, "sources": [], "blocked": True}

        results = self.retrieve(checked)
        context = self.build_context(results)

        if not context:
            return {
                "answer": "I could not find relevant information in the supplied PDF.",
                "sources": [],
                "blocked": False,
            }

        draft = generate_answer(checked, context)
        answer = draft.get("answer", "")
        citations = draft.get("citations", [])

        review = critique(checked, answer, context)

        if review.get("needs_revision", False):
            answer = revise(checked, answer, context, review)

        output_ok, answer = validate_output(answer)

        source_payload = [
            {
                "page": r["page"],
                "chunk_id": r["chunk_id"],
                "vector_score": round(r["vector_score"], 5),
                "graph_score": round(r["graph_score"], 5),
            }
            for r in results[:5]
        ]

        citation_ok, citation_answer = enforce_citations(
            answer,
            source_payload,
        )

        if not output_ok or not citation_ok:
            final_answer = (
                "I could not safely validate the generated answer against "
                "the supplied document."
            )
        else:
            final_answer = citation_answer

        return {
            "answer": final_answer,
            "sources": source_payload,
            "self_rag_review": review,
            "citations": citations,
            "blocked": False,
        }
