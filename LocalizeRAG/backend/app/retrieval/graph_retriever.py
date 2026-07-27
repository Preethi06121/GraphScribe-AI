import logging

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.graph.entity_extractor import EntityExtractor
from app.graph.graph_schema import RULE_BASED_TECH_TERMS
from app.graph.neo4j_service import Neo4jService
from app.schemas.retrieval import GraphHit, ProcessedQuery

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieves connected entities and relationships from Neo4j."""

    def __init__(
        self,
        neo4j_service: Neo4jService,
        entity_extractor: EntityExtractor,
        top_k: int = 5,
    ) -> None:
        self._neo4j_service = neo4j_service
        self._entity_extractor = entity_extractor
        self._top_k = top_k

    def retrieve(self, processed_query: ProcessedQuery, top_k: int | None = None) -> list[GraphHit]:
        k = top_k if top_k is not None else self._top_k
        entity_names = self._extract_query_entities(processed_query)

        if not entity_names:
            logger.info("No entities extracted from query; skipping graph retrieval")
            return []

        if not self._neo4j_service.is_available():
            logger.warning("Neo4j unavailable; returning empty graph hits")
            return []

        try:
            records = self._neo4j_service.search_by_entity_names(entity_names, limit=k * 10)
        except (ServiceUnavailable, Neo4jError, OSError) as exc:
            logger.warning("Graph retrieval failed: %s", exc)
            return []

        hits = self._score_records(records, entity_names)
        hits.sort(key=lambda hit: hit.score, reverse=True)
        trimmed = hits[:k]
        logger.info("Graph retriever returned %d hit(s)", len(trimmed))
        return trimmed

    def _extract_query_entities(self, processed_query: ProcessedQuery) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()

        extracted = self._entity_extractor.extract_from_text(
            processed_query.original,
            document_id="query",
            document_name="query",
            page_number=0,
        )
        for entity in extracted:
            key = entity.entity_name.lower()
            if key not in seen:
                seen.add(key)
                names.append(entity.entity_name)

        for keyword in processed_query.keywords:
            if keyword.lower() in RULE_BASED_TECH_TERMS and keyword.lower() not in seen:
                seen.add(keyword.lower())
                names.append(keyword)

            for term in RULE_BASED_TECH_TERMS:
                if term in processed_query.normalized and term not in seen:
                    seen.add(term)
                    names.append(term)

        for keyword in processed_query.keywords:
            if len(keyword) >= 3 and keyword not in seen:
                seen.add(keyword)
                names.append(keyword)

        return names

    def _score_records(self, records: list[dict], query_entities: list[str]) -> list[GraphHit]:
        query_lower = {name.lower() for name in query_entities}
        hits: list[GraphHit] = []
        seen: set[tuple[str, str, str | None, str | None]] = set()

        for record in records:
            entity_name = record.get("entity_name") or ""
            if not entity_name:
                continue

            key = (
                entity_name,
                record.get("document_id") or "",
                record.get("connected_entity"),
                record.get("relationship_type"),
            )
            if key in seen:
                continue
            seen.add(key)

            score = 0.5
            if entity_name.lower() in query_lower:
                score += 0.3
            if record.get("connected_entity"):
                score += 0.15
            if record.get("relationship_type"):
                score += 0.05
            score = min(score, 1.0)

            hits.append(
                GraphHit(
                    entity_name=entity_name,
                    entity_type=str(record.get("entity_type") or "UNKNOWN"),
                    connected_entity=record.get("connected_entity"),
                    connected_type=record.get("connected_type"),
                    relationship_type=record.get("relationship_type"),
                    document_id=str(record.get("document_id") or ""),
                    document_name=str(record.get("document_name") or ""),
                    page_number=int(record.get("page_number") or 0),
                    score=score,
                )
            )

        return hits
