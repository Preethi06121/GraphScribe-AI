from __future__ import annotations

import asyncio
import time
from typing import List

from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.retrieval import HybridContext
from app.research.strategies.base import RetrievalStrategy, StrategyResult
from app.core.deps import get_hybrid_retriever


class HybridStrategy(RetrievalStrategy):
    name = "hybrid"

    def __init__(self, hybrid_retriever: HybridRetriever | None = None):
        self._hybrid_retriever = hybrid_retriever or get_hybrid_retriever()

    async def run(self, query: str, k: int) -> StrategyResult:
        start = time.monotonic()
        # HybridRetriever.retrieve is async
        context: HybridContext = await self._hybrid_retriever.retrieve(query)
        end = time.monotonic()

        retrieved_items = []
        citations = []

        # context.documents are ContextDocument models
        for doc in context.documents:
            retrieved_items.append(
                {
                    "document_id": doc.document_id,
                    "document_name": doc.document_name,
                    "page": doc.page,
                    "score": doc.score,
                    "snippet": doc.chunk[:400],
                    "source": "vector",
                }
            )
            if doc.document_id:
                citations.append(doc.document_id)

        # context.graph are ContextGraphItem models
        entity_names = set()
        for g in context.graph:
            retrieved_items.append(
                {
                    "entity_name": g.entity_name,
                    "entity_type": g.entity_type,
                    "connected_entity": g.connected_entity,
                    "relationship_type": g.relationship_type,
                    "document_id": g.document_id,
                    "document_name": g.document_name,
                    "page": g.page_number,
                    "score": g.score,
                    "source": "graph",
                }
            )
            if g.document_id:
                citations.append(g.document_id)
            if g.entity_name:
                entity_names.add(g.entity_name)

        unique_docs = list(dict.fromkeys(citations))

        return StrategyResult(
            strategy=self.name,
            latency_ms=(end - start) * 1000,
            documents_retrieved=len(unique_docs),
            graph_entities=len(entity_names) if entity_names else None,
            citations=unique_docs,
            retrieved_items=retrieved_items,
        )
