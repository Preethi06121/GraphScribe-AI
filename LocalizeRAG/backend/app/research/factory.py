import logging

from app.research.strategies.base import RetrievalStrategy
from app.research.strategies.graph_strategy import GraphStrategy
from app.research.strategies.hybrid_strategy import HybridStrategy
from app.research.strategies.vector_strategy import VectorStrategy
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_processor import QueryProcessor
from app.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class StrategyFactory:
    """Factory for creating retrieval strategy instances."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        hybrid_retriever: HybridRetriever,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._graph_retriever = graph_retriever
        self._hybrid_retriever = hybrid_retriever
        self._query_processor = query_processor or QueryProcessor()

    def get_strategy(self, strategy_name: str) -> RetrievalStrategy:
        name_clean = strategy_name.strip().lower()

        if name_clean == "vector":
            return VectorStrategy(
                vector_retriever=self._vector_retriever,
                query_processor=self._query_processor,
            )
        elif name_clean == "graph":
            return GraphStrategy(
                graph_retriever=self._graph_retriever,
                query_processor=self._query_processor,
            )
        elif name_clean == "hybrid":
            return HybridStrategy(hybrid_retriever=self._hybrid_retriever)
        else:
            raise ValueError(
                f"Unknown retrieval strategy: '{strategy_name}'. "
                f"Supported strategies are: 'vector', 'graph', 'hybrid'."
            )

    @staticmethod
    def get_supported_strategies() -> list[str]:
        return ["vector", "graph", "hybrid"]
