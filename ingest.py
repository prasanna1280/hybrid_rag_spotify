import argparse

from graph_store import KnowledgeGraph
from llm import json_chat
from pdf_processor import extract_pdf, section_chunk
from vector_store import VectorStore


ENTITY_SYSTEM = """
You are building a knowledge graph from a technical architecture document.

Extract only entities and explicit relationships supported by the text.

Return JSON:
{
  "entities": [
    {"name": "PostgreSQL", "type": "Technology"},
    {"name": "User Profile Service", "type": "Service"}
  ],
  "relationships": [
    {"source": "User Profile Service", "relation": "STORES_DATA_IN", "target": "PostgreSQL"}
  ]
}

Do not invent relationships.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--clear-graph", action="store_true")
    args = parser.parse_args()

    pages = extract_pdf(args.pdf)
    chunks = section_chunk(pages)

    print(f"Extracted {len(pages)} pages and {len(chunks)} chunks.")

    vector = VectorStore()
    vector.build(chunks)
    print("FAISS vector index created.")

    graph = KnowledgeGraph()

    if args.clear_graph:
        graph.clear()
        print("Neo4j graph cleared.")

    for i, chunk in enumerate(chunks, start=1):
        graph.upsert_chunk(chunk)

        # One LLM call per chunk. For large PDFs, batch chunks to reduce cost.
        extracted = json_chat(ENTITY_SYSTEM, chunk["text"])

        for entity in extracted.get("entities", []):
            if entity.get("name"):
                graph.add_entity(chunk["chunk_id"], entity)

        for rel in extracted.get("relationships", []):
            if rel.get("source") and rel.get("target"):
                graph.add_relationship(
                    rel["source"],
                    rel.get("relation", "RELATED_TO"),
                    rel["target"],
                )

        print(f"Processed chunk {i}/{len(chunks)}")

    print("Ingestion completed.")
    print("Neo4j stats:", graph.stats())
    graph.close()


if __name__ == "__main__":
    main()
