from __future__ import annotations

from typing import Optional

from app.research.strategies.vector_strategy import VectorStrategy
from app.research.strategies.graph_strategy import GraphStrategy
from app.research.strategies.hybrid_strategy import HybridStrategy


class StrategyFactory:
    @staticmethod
    def create(name: str, **kwargs) -> Optional[object]:
        key = name.strip().lower()
        if key == "vector":
            return VectorStrategy(**kwargs)
        if key == "graph":
            return GraphStrategy(**kwargs)
        if key == "hybrid":
            return HybridStrategy(**kwargs)
        return None
