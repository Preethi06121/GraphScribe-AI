import logging
from datetime import datetime, timezone

from app.research.factory import StrategyFactory
from app.research.schemas import (
    ResearchComparisonRequest,
    ResearchComparisonResponse,
    StrategyResult,
)

logger = logging.getLogger(__name__)


class ComparisonService:
    """Orchestrates sequential retrieval strategy execution and comparative analysis."""

    def __init__(self, factory: StrategyFactory) -> None:
        self._factory = factory

    async def compare(
        self,
        request: ResearchComparisonRequest,
    ) -> ResearchComparisonResponse:
        logger.info(
            "Starting comparative evaluation for query: '%s' (strategies=%s, k=%d)",
            request.query,
            request.strategies,
            request.k,
        )

        results: list[StrategyResult] = []

        # Execute strategies sequentially
        for strat_name in request.strategies:
            try:
                strategy_impl = self._factory.get_strategy(strat_name)
                result = await strategy_impl.run(
                    query=request.query,
                    k=request.k,
                    include_items=request.include_items,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Strategy '%s' failed during execution: %s",
                    strat_name,
                    exc,
                    exc_info=True,
                )
                results.append(
                    StrategyResult(
                        strategy=strat_name,
                        latency_ms=0.0,
                        documents_retrieved=0,
                        graph_entities=0,
                        citations=[],
                        retrieved_items=[],
                        success=False,
                        error=str(exc),
                    )
                )

        best_strategy, reason = self._evaluate_best_strategy(results)
        utc_timestamp = datetime.now(timezone.utc).isoformat()

        return ResearchComparisonResponse(
            query=request.query,
            results=results,
            best_strategy=best_strategy,
            reason=reason,
            timestamp=utc_timestamp,
        )

    @staticmethod
    def _evaluate_best_strategy(results: list[StrategyResult]) -> tuple[str, str]:
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return "none", "All strategy executions failed or encountered errors."

        # Selection criteria:
        # 1. Highest documents_retrieved
        # 2. Lowest latency if tied (or graph_entities if documents_retrieved == 0)
        best = max(
            successful_results,
            key=lambda r: (
                r.documents_retrieved,
                -r.latency_ms if r.documents_retrieved > 0 else (r.graph_entities, -r.latency_ms),
            ),
        )

        total_items = best.documents_retrieved + best.graph_entities
        if best.documents_retrieved > 0:
            reason = (
                f"Selected '{best.strategy}' as best strategy because it retrieved "
                f"{best.documents_retrieved} document(s) with the lowest latency ({best.latency_ms:.2f} ms)."
            )
        elif total_items > 0:
            reason = (
                f"Selected '{best.strategy}' as best strategy because it retrieved "
                f"{best.graph_entities} graph entit(y/ies) with latency ({best.latency_ms:.2f} ms)."
            )
        else:
            reason = (
                f"Selected '{best.strategy}' as default fallback with lowest latency "
                f"({best.latency_ms:.2f} ms) when no results were retrieved."
            )

        return best.strategy, reason
