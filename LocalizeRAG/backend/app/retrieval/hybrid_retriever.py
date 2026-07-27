import asyncio
import hashlib
import logging
from collections import OrderedDict

from app.retrieval.context_fusion import ContextFusion
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.query_processor import QueryProcessor
from app.retrieval.ranking import RankingService
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.retrieval import HybridContext, ProcessedQuery

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrates vector + graph retrieval into a single ranked context."""

    def __init__(
        self,
        query_processor: QueryProcessor,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        ranking_service: RankingService,
        context_fusion: ContextFusion,
        cache_size: int = 128,
        top_k: int = 5,
    ) -> None:
        self._query_processor = query_processor
        self._vector_retriever = vector_retriever
        self._graph_retriever = graph_retriever
        self._ranking_service = ranking_service
        self._context_fusion = context_fusion
        self._cache_size = cache_size
        self._top_k = top_k
        self._cache: OrderedDict[str, HybridContext] = OrderedDict()

    async def retrieve(self, query: str) -> HybridContext:
        cache_key = self._cache_key(query)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.info("Returning cached hybrid context for query")
            return cached

        processed = await asyncio.to_thread(self._query_processor.process, query)
        context = await self._retrieve_processed(processed)
        self._set_cached(cache_key, context)
        return context

    async def _retrieve_processed(self, processed: ProcessedQuery) -> HybridContext:
        logger.info("Starting hybrid retrieval for query: %s", processed.normalized)

        vector_hits, graph_hits = await asyncio.gather(
            asyncio.to_thread(self._vector_retriever.retrieve, processed, self._top_k),
            asyncio.to_thread(self._graph_retriever.retrieve, processed, self._top_k),
        )

        ranked = await asyncio.to_thread(
            self._ranking_service.rank,
            vector_hits,
            graph_hits,
        )
        context = await asyncio.to_thread(
            self._context_fusion.fuse,
            processed,
            vector_hits,
            graph_hits,
            ranked,
            self._ranking_service.vector_weight,
            self._ranking_service.graph_weight,
            self._top_k,
            self._top_k,
        )
        logger.info("Hybrid retrieval complete")
        return context

    @staticmethod
    def _cache_key(query: str) -> str:
        normalized = " ".join(query.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> HybridContext | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _set_cached(self, key: str, context: HybridContext) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = context
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        self._cache.clear()
