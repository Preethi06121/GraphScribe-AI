import asyncio
import logging
import time

from app.research.schemas import StrategyResult
from app.research.strategies.base import RetrievalStrategy
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class GraphStrategy(RetrievalStrategy):
    """Retrieval strategy wrapping the existing GraphRetriever."""

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        self._graph_retriever = graph_retriever
        self._query_processor = query_processor or QueryProcessor()

    @property
    def name(self) -> str:
        return "graph"

    async def run(
        self,
        query: str,
        k: int = 5,
        include_items: bool = False,
    ) -> StrategyResult:
        start_time = time.perf_counter()
        logger.info("Executing GraphStrategy for query: %s (k=%d)", query, k)

        processed = await asyncio.to_thread(self._query_processor.process, query)
        hits = await asyncio.to_thread(self._graph_retriever.retrieve, processed, k)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        citations = []
        for hit in hits:
            if hit.connected_entity and hit.relationship_type:
                citations.append(
                    f"{hit.entity_name} [{hit.relationship_type}] {hit.connected_entity}"
                )
            else:
                citations.append(f"{hit.entity_name} ({hit.entity_type})")

        items = []
        if include_items:
            items = [hit.model_dump() for hit in hits]

        return StrategyResult(
            strategy=self.name,
            latency_ms=elapsed_ms,
            documents_retrieved=0,
            graph_entities=len(hits),
            citations=citations,
            retrieved_items=items,
            success=True,
        )

    async def retrieve_context(self, query: str, k: int = 5):
        from app.schemas.retrieval import (
            Citation,
            ContextGraphItem,
            HybridContext,
            HybridContextMetadata,
        )

        processed = await asyncio.to_thread(self._query_processor.process, query)
        hits = await asyncio.to_thread(self._graph_retriever.retrieve, processed, k)

        graph_items = [
            ContextGraphItem(
                entity_name=hit.entity_name,
                entity_type=hit.entity_type,
                connected_entity=hit.connected_entity,
                relationship_type=hit.relationship_type,
                document_id=hit.document_id,
                document_name=hit.document_name,
                page_number=hit.page_number,
                score=hit.score,
                source="graph",
            )
            for hit in hits
        ]
        citations = [
            Citation(
                document_id=hit.document_id,
                document_name=hit.document_name or "Graph",
                page=hit.page_number,
                source="graph",
                reference=f"{hit.entity_name}-{hit.relationship_type or 'node'}",
            )
            for hit in hits
        ]
        return HybridContext(
            documents=[],
            graph=graph_items,
            citations=citations,
            metadata=HybridContextMetadata(
                query=query,
                normalized_query=processed.normalized,
                keywords=processed.keywords,
                vector_hits=0,
                graph_hits=len(graph_items),
                vector_weight=0.0,
                graph_weight=1.0,
            ),
        )
