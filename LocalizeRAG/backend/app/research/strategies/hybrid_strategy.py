import logging
import time

from app.research.schemas import StrategyResult
from app.research.strategies.base import RetrievalStrategy
from app.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class HybridStrategy(RetrievalStrategy):
    """Retrieval strategy wrapping the existing HybridRetriever."""

    def __init__(self, hybrid_retriever: HybridRetriever) -> None:
        self._hybrid_retriever = hybrid_retriever

    @property
    def name(self) -> str:
        return "hybrid"

    async def run(
        self,
        query: str,
        k: int = 5,
        include_items: bool = False,
    ) -> StrategyResult:
        start_time = time.perf_counter()
        logger.info("Executing HybridStrategy for query: %s (k=%d)", query, k)

        context = await self._hybrid_retriever.retrieve(query)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        items = []
        if include_items:
            doc_items = [doc.model_dump() for doc in context.documents]
            graph_items = [g.model_dump() for g in context.graph]
            items = doc_items + graph_items

        citations = [
            c if isinstance(c, str) else (f"{c.document_name} (Page {c.page})" if c.page else f"{c.document_name}")
            for c in context.citations
        ]

        return StrategyResult(
            strategy=self.name,
            latency_ms=elapsed_ms,
            documents_retrieved=len(context.documents),
            graph_entities=len(context.graph),
            citations=citations,
            retrieved_items=items,
            success=True,
        )
