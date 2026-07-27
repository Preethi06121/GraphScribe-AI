from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from app.research.factory import StrategyFactory
from app.research.schemas import ResearchComparisonRequest, ResearchComparisonResponse
from app.research.strategies.base import StrategyResult


class ComparisonService:
    """Sequentially runs retrieval strategies and returns structured metrics.

    Executes strategies in sequence (order provided by request) and collects
    metrics for each. If a strategy fails, records a structured error and
    continues with the remaining strategies.
    """

    def __init__(self) -> None:
        pass

    async def compare(self, request: ResearchComparisonRequest) -> ResearchComparisonResponse:
        query = request.query
        strategies = request.strategies or ["vector", "graph", "hybrid"]
        k = request.k if request.k is not None else 5
        include_items = bool(request.include_items)

        results: Dict[str, Any] = {}

        for name in strategies:
            strategy = StrategyFactory.create(name)
            if strategy is None:
                results[name] = {"error": f"Unknown strategy: {name}"}
                continue

            try:
                # Each strategy is responsible for measuring its own internal
                # retrieval latency. We still measure end-to-end time here for
                # defensive purposes but prefer the strategy-reported latency_ms
                # when present.
                start = time.monotonic()
                result: StrategyResult = await strategy.run(query, k)
                end = time.monotonic()

                latency_ms = float(getattr(result, "latency_ms", (end - start) * 1000))

                metrics = {
                    "strategy": result.strategy,
                    "latency_ms": round(latency_ms, 3),
                    "documents_retrieved": int(getattr(result, "documents_retrieved", 0)),
                    "graph_entities": int(result.graph_entities) if result.graph_entities is not None else None,
                    "citations": list(getattr(result, "citations", [])),
                    "retrieved_items": list(result.retrieved_items) if include_items else [],
                }
                results[name] = metrics
            except Exception as exc:  # pragma: no cover - capture and continue
                results[name] = {"error": str(exc)}

        # Choose best strategy heuristic: highest documents_retrieved then lowest latency
        best_strategy = None
        reason = None
        try:
            sortable = []
            for n, v in results.items():
                if not isinstance(v, dict) or v.get("error"):
                    continue
                docs = v.get("documents_retrieved", 0)
                latency = v.get("latency_ms", float("inf"))
                # We want to sort by docs desc, latency asc
                sortable.append((docs, -latency, n))
            if sortable:
                sortable.sort(reverse=True)
                best_strategy = sortable[0][2]
                best_metrics = results[best_strategy]
                reason = (
                    f"Selected {best_strategy} with documents_retrieved={best_metrics.get('documents_retrieved')} "
                    f"and latency_ms={best_metrics.get('latency_ms')}"
                )
        except Exception:
            best_strategy = None
            reason = None

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        response = {
            "query": query,
            "timestamp": timestamp,
            "results": results,
            "best_strategy": best_strategy,
            "reason": reason,
            "meta": {"k": k, "run_id": str(uuid.uuid4()), "timestamp_utc": timestamp},
        }

        # Validate via Pydantic model
        return ResearchComparisonResponse.parse_obj(response)
