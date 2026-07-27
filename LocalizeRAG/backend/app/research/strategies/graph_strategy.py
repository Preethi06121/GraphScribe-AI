from __future__ import annotations

import asyncio
import time
from typing import List

from app.retrieval.query_processor import QueryProcessor
from app.schemas.retrieval import ProcessedQuery, GraphHit
from app.research.strategies.base import RetrievalStrategy, StrategyResult
from app.core.deps import get_graph_retriever, get_query_processor


class GraphStrategy(RetrievalStrategy):
    name = "graph"

    def __init__(self, graph_retriever=None, query_processor: QueryProcessor | None = None):
        self._graph_retriever = graph_retriever or get_graph_retriever()
        self._query_processor = query_processor or get_query_processor()

    async def run(self, query: str, k: int) -> StrategyResult:
        processed: ProcessedQuery = await asyncio.to_thread(self._query_processor.process, query)

        start = time.monotonic()
        hits: List[GraphHit] = await asyncio.to_thread(self._graph_retriever.retrieve, processed, k)
        end = time.monotonic()

        retrieved_items = []
        citations = []
        entity_names = set()
        for h in hits:
            retrieved_items.append(
                {
                    "entity_name": h.entity_name,
                    "entity_type": h.entity_type,
                    "connected_entity": h.connected_entity,
                    "relationship_type": h.relationship_type,
                    "document_id": h.document_id,
                    "document_name": h.document_name,
                    "page": h.page_number,
                    "score": h.score,
                    "source": "graph",
                }
            )
            if h.document_id:
                citations.append(h.document_id)
            if h.entity_name:
                entity_names.add(h.entity_name)

        return StrategyResult(
            strategy=self.name,
            latency_ms=(end - start) * 1000,
            documents_retrieved=len({c for c in citations}),
            graph_entities=len(entity_names),
            citations=list(dict.fromkeys(citations)),
            retrieved_items=retrieved_items,
        )
