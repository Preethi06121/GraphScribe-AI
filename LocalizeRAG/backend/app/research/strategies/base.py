from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class StrategyResult(BaseModel):
    strategy: str
    latency_ms: float
    documents_retrieved: int
    graph_entities: int | None = None
    citations: list[str] = Field(default_factory=list)
    retrieved_items: list[dict] = Field(default_factory=list)


class RetrievalStrategy(ABC):
    name: str

    @abstractmethod
    async def run(self, query: str, k: int) -> StrategyResult:
        """Execute retrieval for the given query and return a StrategyResult.

        Implementations MUST be async and may use asyncio.to_thread to call
        synchronous retrieval functions in the codebase.
        """
        raise NotImplementedError
