from typing import Any

from pydantic import BaseModel, Field, field_validator


class StrategyResult(BaseModel):
    """Result of a single retrieval strategy execution."""

    strategy: str
    latency_ms: float = 0.0
    documents_retrieved: int = 0
    graph_entities: int = 0
    citations: list[str] = Field(default_factory=list)
    retrieved_items: list[dict[str, Any]] = Field(default_factory=list)
    success: bool = True
    error: str | None = None


class ResearchComparisonRequest(BaseModel):
    """Request payload for comparing retrieval strategies."""

    query: str
    strategies: list[str] = Field(default_factory=list)
    k: int = 5
    include_items: bool = False

    @field_validator("strategies", mode="after")
    @classmethod
    def set_default_strategies(cls, v: list[str]) -> list[str]:
        if not v:
            return ["vector", "graph", "hybrid"]
        return v


class ResearchComparisonResponse(BaseModel):
    """Response payload containing comparison metrics across strategies."""

    query: str
    results: list[StrategyResult] = Field(default_factory=list)
    best_strategy: str
    reason: str
    timestamp: str
