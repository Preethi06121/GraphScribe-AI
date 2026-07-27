from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ResearchComparisonRequest(BaseModel):
    query: str
    strategies: List[str] = Field(default_factory=lambda: ["vector", "graph", "hybrid"])
    k: int = 5
    include_items: bool = False


class StrategyMetrics(BaseModel):
    strategy: str
    latency_ms: float
    documents_retrieved: int
    graph_entities: int | None = None
    citations: List[str] = Field(default_factory=list)
    retrieved_items: List[Dict[str, Any]] = Field(default_factory=list)


class ResearchComparisonResponse(BaseModel):
    query: str
    timestamp: str
    results: Dict[str, Any]
    best_strategy: str | None = None
    reason: str | None = None
    meta: Dict[str, Any]
