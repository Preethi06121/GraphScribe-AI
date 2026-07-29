from abc import ABC, abstractmethod

from app.research.schemas import StrategyResult


class RetrievalStrategy(ABC):
    """Abstract base class for all retrieval strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the identifier name of the strategy."""

    @abstractmethod
    async def run(
        self,
        query: str,
        k: int = 5,
        include_items: bool = False,
    ) -> StrategyResult:
        """Executes retrieval strategy and returns structured StrategyResult."""


# Alias for backward/forward compatibility
BaseRetrievalStrategy = RetrievalStrategy
