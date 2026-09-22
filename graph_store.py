import json
from neo4j import GraphDatabase

from config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)


class KnowledgeGraph:
    def __init__(self):
        if not (NEO4J_URI and NEO4J_PASSWORD):
            raise RuntimeError(
                "Neo4j configuration is missing. Set NEO4J_URI and NEO4J_PASSWORD."
            )

        self.database = (NEO4J_DATABASE or "neo4j").strip() or "neo4j"
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )

    def close(self):
        self.driver.close()

    def clear(self):
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def upsert_chunk(self, chunk):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (c:Chunk {id: $id})
                SET c.page = $page, c.text = $text
                """,
                id=chunk["chunk_id"],
                page=chunk["page"],
                text=chunk["text"],
            )

    def add_entity(self, chunk_id, entity):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (e:Entity {name: $name, type: $type})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                chunk_id=chunk_id,
                name=entity["name"],
                type=entity.get("type", "Entity"),
            )

    def add_relationship(self, source, relation, target):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (a:Entity {name: $source})
                MERGE (b:Entity {name: $target})
                MERGE (a)-[r:RELATED_TO {type: $relation}]->(b)
                """,
                source=source,
                target=target,
                relation=relation,
            )

    def search_entities(self, entity_names, limit=5):
        if not entity_names:
            return []

        with self.driver.session(database=self.database) as session:
            rows = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE any(name IN $names WHERE toLower(e.name) CONTAINS toLower(name))
                OPTIONAL MATCH (e)-[r:RELATED_TO]->(other:Entity)
                RETURN c.id AS chunk_id,
                       c.page AS page,
                       c.text AS text,
                       e.name AS entity,
                       e.type AS entity_type,
                       collect({
                           relation: r.type,
                           target: other.name
                       }) AS relationships
                LIMIT $limit
                """,
                names=entity_names,
                limit=limit,
            )
            return [dict(row) for row in rows]

    def stats(self):
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count"
            )
            return [dict(r) for r in result]
