from __future__ import annotations

import asyncio
import time

from typing import List

from app.retrieval.query_processor import QueryProcessor
from app.schemas.retrieval import ProcessedQuery, VectorHit
from app.research.strategies.base import RetrievalStrategy, StrategyResult
from app.core.deps import get_vector_retriever, get_query_processor


class VectorStrategy(RetrievalStrategy):
    name = "vector"

    def __init__(self, vector_retriever=None, query_processor: QueryProcessor | None = None):
        # Use DI singletons when not provided
        self._vector_retriever = vector_retriever or get_vector_retriever()
        self._query_processor = query_processor or get_query_processor()

    async def run(self, query: str, k: int) -> StrategyResult:
        # Process query synchronously via to_thread to avoid blocking
        processed: ProcessedQuery = await asyncio.to_thread(self._query_processor.process, query)

        start = time.monotonic()
        hits: List[VectorHit] = await asyncio.to_thread(self._vector_retriever.retrieve, processed, k)
        end = time.monotonic()

        retrieved_items = []
        citations = []
        for h in hits:
            retrieved_items.append(
                {
                    "document_id": h.document_id,
                    "document_name": h.document_name,
                    "page": h.page,
                    "score": h.score,
                    "snippet": h.chunk[:400],
                    "source": h.source or "vector",
                }
            )
            if h.document_id:
                citations.append(h.document_id)

        return StrategyResult(
            strategy=self.name,
            latency_ms=(end - start) * 1000,
            documents_retrieved=len(hits),
            graph_entities=None,
            citations=list(dict.fromkeys(citations)),
            retrieved_items=retrieved_items,
        )
