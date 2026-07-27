import logging
import re

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.graph.graph_schema import ENTITY_CONSTRAINT_QUERY, ENTITY_LABEL
from app.schemas.graph import (
    DocumentGraphNode,
    DocumentGraphRelationship,
    DocumentGraphResponse,
    DocumentGraphStatistics,
    GraphEntity,
    GraphRelationship,
    GraphStatistics,
)

logger = logging.getLogger(__name__)


class Neo4jService:
    """Manages Neo4j connections and graph persistence."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = None
        self._constraints_initialized = False

    def _get_driver(self):
        if self._driver is None:
            logger.info("Connecting to Neo4j at %s", self._uri)
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def is_available(self) -> bool:
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            return True
        except (ServiceUnavailable, Neo4jError, OSError) as exc:
            logger.warning("Neo4j is unavailable: %s", exc)
            return False

    def ensure_constraints(self) -> None:
        if self._constraints_initialized:
            return
        try:
            with self._get_driver().session(database=self._database) as session:
                session.run(ENTITY_CONSTRAINT_QUERY)
            self._constraints_initialized = True
            logger.info("Neo4j constraints initialized")
        except (ServiceUnavailable, Neo4jError, OSError) as exc:
            logger.warning("Failed to initialize Neo4j constraints: %s", exc)
            raise

    def merge_entities(self, entities: list[GraphEntity]) -> int:
        if not entities:
            return 0

        self.ensure_constraints()
        query = f"""
        UNWIND $entities AS entity
        MERGE (n:{ENTITY_LABEL} {{
            entity_name: entity.entity_name,
            entity_type: entity.entity_type,
            document_id: entity.document_id,
            page_number: entity.page_number
        }})
        SET n.document_name = entity.document_name
        RETURN count(n) AS count
        """

        entity_data = [entity.model_dump() for entity in entities]
        with self._get_driver().session(database=self._database) as session:
            result = session.run(query, entities=entity_data)
            record = result.single()
            return record["count"] if record else 0

    def merge_relationships(self, relationships: list[GraphRelationship]) -> int:
        if not relationships:
            return 0

        count = 0
        with self._get_driver().session(database=self._database) as session:
            for rel in relationships:
                rel_type = self._sanitize_rel_type(rel.relationship_type)
                query = f"""
                MATCH (source:{ENTITY_LABEL} {{
                    entity_name: $source_entity,
                    entity_type: $source_type,
                    document_id: $document_id,
                    page_number: $page_number
                }})
                MATCH (target:{ENTITY_LABEL} {{
                    entity_name: $target_entity,
                    entity_type: $target_type,
                    document_id: $document_id,
                    page_number: $page_number
                }})
                MERGE (source)-[r:{rel_type}]->(target)
                SET r.document_id = $document_id,
                    r.source_document = $source_document,
                    r.page_number = $page_number
                RETURN count(r) AS count
                """
                result = session.run(
                    query,
                    source_entity=rel.source_entity,
                    source_type=rel.source_type,
                    target_entity=rel.target_entity,
                    target_type=rel.target_type,
                    document_id=rel.document_id,
                    source_document=rel.source_document,
                    page_number=rel.page_number,
                )
                record = result.single()
                count += record["count"] if record else 0

        return count

    def get_statistics(self) -> GraphStatistics:
        query = """
        MATCH (n:Entity)
        WITH count(n) AS node_count,
             collect(DISTINCT n.entity_type) AS entity_types,
             count(DISTINCT n.document_id) AS document_count
        OPTIONAL MATCH ()-[r]->()
        RETURN node_count, entity_types, document_count, count(r) AS rel_count
        """

        with self._get_driver().session(database=self._database) as session:
            record = session.run(query).single()

        if not record:
            return GraphStatistics(
                nodes=0,
                relationships=0,
                entity_types=[],
                documents=0,
            )

        return GraphStatistics(
            nodes=record["node_count"],
            relationships=record["rel_count"],
            entity_types=sorted(record["entity_types"] or []),
            documents=record["document_count"],
        )

    def get_document_graph(self, document_id: str) -> DocumentGraphResponse:
        nodes_query = """
        MATCH (n:Entity {document_id: $document_id})
        RETURN n.entity_name AS entity_name,
               n.entity_type AS entity_type,
               n.document_id AS document_id,
               n.document_name AS document_name,
               n.page_number AS page_number
        ORDER BY n.page_number, n.entity_name
        """

        rels_query = """
        MATCH (source:Entity {document_id: $document_id})-[r]->(target:Entity {document_id: $document_id})
        RETURN source.entity_name AS source_entity,
               source.entity_type AS source_type,
               type(r) AS relationship_type,
               target.entity_name AS target_entity,
               target.entity_type AS target_type,
               r.document_id AS document_id,
               r.source_document AS source_document,
               r.page_number AS page_number
        """

        with self._get_driver().session(database=self._database) as session:
            node_records = session.run(nodes_query, document_id=document_id).data()
            rel_records = session.run(rels_query, document_id=document_id).data()

        nodes = [DocumentGraphNode(**record) for record in node_records]
        relationships = [DocumentGraphRelationship(**record) for record in rel_records]
        entity_types = sorted({node.entity_type for node in nodes})

        return DocumentGraphResponse(
            document_id=document_id,
            nodes=nodes,
            relationships=relationships,
            statistics=DocumentGraphStatistics(
                nodes=len(nodes),
                relationships=len(relationships),
                entity_types=entity_types,
            ),
        )

    def search_by_entity_names(self, entity_names: list[str], limit: int = 50) -> list[dict]:
        """Find matching entities and their connected neighbors for retrieval."""
        if not entity_names:
            return []

        lowered = [name.lower() for name in entity_names]
        query = f"""
        MATCH (n:{ENTITY_LABEL})
        WHERE toLower(n.entity_name) IN $entity_names
        OPTIONAL MATCH (n)-[r]-(m:{ENTITY_LABEL})
        RETURN n.entity_name AS entity_name,
               n.entity_type AS entity_type,
               n.document_id AS document_id,
               n.document_name AS document_name,
               n.page_number AS page_number,
               m.entity_name AS connected_entity,
               m.entity_type AS connected_type,
               type(r) AS relationship_type
        LIMIT $limit
        """

        with self._get_driver().session(database=self._database) as session:
            records = session.run(query, entity_names=lowered, limit=limit).data()

        logger.info(
            "Neo4j entity search for %d name(s) returned %d record(s)",
            len(entity_names),
            len(records),
        )
        return records

    @staticmethod
    def _sanitize_rel_type(rel_type: str) -> str:
        sanitized = re.sub(r"[^A-Z0-9_]", "_", rel_type.upper())
        if not sanitized:
            return "RELATED_TO"
        if sanitized[0].isdigit():
            return f"REL_{sanitized}"
        return sanitized
