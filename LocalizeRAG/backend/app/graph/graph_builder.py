import asyncio
import logging
from pathlib import Path

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.graph.entity_extractor import EntityExtractor
from app.graph.relationship_extractor import RelationshipExtractor
from app.graph.neo4j_service import Neo4jService
from app.rag.document_loader import PDFDocumentLoader
from app.schemas.document import PageContent

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds a knowledge graph from document pages and persists to Neo4j."""

    def __init__(
        self,
        document_loader: PDFDocumentLoader,
        entity_extractor: EntityExtractor,
        relationship_extractor: RelationshipExtractor,
        neo4j_service: Neo4jService,
    ) -> None:
        self._document_loader = document_loader
        self._entity_extractor = entity_extractor
        self._relationship_extractor = relationship_extractor
        self._neo4j_service = neo4j_service

    async def build_from_file(
        self,
        file_path: str | Path,
        document_id: str,
        document_name: str,
    ) -> None:
        path = Path(file_path)
        pages = await asyncio.to_thread(self._document_loader.load, path)
        await self.build_from_pages(pages, document_id, document_name)

    async def build_from_pages(
        self,
        pages: list[PageContent],
        document_id: str,
        document_name: str,
    ) -> None:
        if not self._neo4j_service.is_available():
            logger.warning(
                "Skipping graph build for %s: Neo4j is unavailable",
                document_name,
            )
            return

        logger.info(
            "Building knowledge graph for %s (id=%s)",
            document_name,
            document_id,
        )

        all_entities = []
        all_relationships = []

        for page in pages:
            entities = await asyncio.to_thread(
                self._entity_extractor.extract_from_text,
                page.text,
                document_id,
                document_name,
                page.page_number,
            )
            if not entities:
                continue

            doc = await asyncio.to_thread(self._entity_extractor.get_nlp_doc, page.text)
            relationships = await asyncio.to_thread(
                self._relationship_extractor.extract,
                page.text,
                entities,
                document_id,
                document_name,
                page.page_number,
                doc,
            )

            all_entities.extend(entities)
            all_relationships.extend(relationships)

        if not all_entities:
            logger.info("No entities found for %s; skipping Neo4j insertion", document_name)
            return

        try:
            await asyncio.to_thread(self._neo4j_service.merge_entities, all_entities)
            await asyncio.to_thread(
                self._neo4j_service.merge_relationships,
                all_relationships,
            )
            logger.info(
                "Graph built for %s: %d node(s), %d relationship(s)",
                document_name,
                len(all_entities),
                len(all_relationships),
            )
        except (ServiceUnavailable, Neo4jError, OSError) as exc:
            logger.warning(
                "Failed to persist graph for %s: %s",
                document_name,
                exc,
            )
