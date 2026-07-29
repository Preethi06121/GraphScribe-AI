import asyncio
import logging
import time

from app.research.schemas import StrategyResult
from app.research.strategies.base import RetrievalStrategy
from app.retrieval.query_processor import QueryProcessor
from app.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class VectorStrategy(RetrievalStrategy):
    """Retrieval strategy wrapping the existing VectorRetriever."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._query_processor = query_processor or QueryProcessor()

    @property
    def name(self) -> str:
        return "vector"

    async def run(
        self,
        query: str,
        k: int = 5,
        include_items: bool = False,
    ) -> StrategyResult:
        start_time = time.perf_counter()
        logger.info("Executing VectorStrategy for query: %s (k=%d)", query, k)

        processed = await asyncio.to_thread(self._query_processor.process, query)
        hits = await asyncio.to_thread(self._vector_retriever.retrieve, processed, k)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        citations = [
            f"{hit.document_name} (Page {hit.page})"
            if hit.document_name
            else f"Doc {hit.document_id} (Page {hit.page})"
            for hit in hits
        ]

        items = []
        if include_items:
            items = [hit.model_dump() for hit in hits]

        return StrategyResult(
            strategy=self.name,
            latency_ms=elapsed_ms,
            documents_retrieved=len(hits),
            graph_entities=0,
            citations=citations,
            retrieved_items=items,
            success=True,
        )
