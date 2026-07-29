import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.research.comparison_service import ComparisonService
from app.research.schemas import ResearchComparisonRequest, StrategyResult


def test_comparison_service_sequential_execution_and_metrics():
    mock_factory = MagicMock()

    vector_strategy = AsyncMock()
    vector_strategy.run.return_value = StrategyResult(
        strategy="vector",
        latency_ms=45.0,
        documents_retrieved=3,
        graph_entities=0,
        citations=["doc1.pdf (Page 1)"],
        retrieved_items=[],
        success=True,
    )

    graph_strategy = AsyncMock()
    graph_strategy.run.return_value = StrategyResult(
        strategy="graph",
        latency_ms=30.0,
        documents_retrieved=0,
        graph_entities=4,
        citations=["EntityA [USES] EntityB"],
        retrieved_items=[],
        success=True,
    )

    hybrid_strategy = AsyncMock()
    hybrid_strategy.run.return_value = StrategyResult(
        strategy="hybrid",
        latency_ms=25.0,
        documents_retrieved=5,
        graph_entities=4,
        citations=["doc1.pdf (Page 1)"],
        retrieved_items=[],
        success=True,
    )

    def get_strat(name):
        if name == "vector":
            return vector_strategy
        elif name == "graph":
            return graph_strategy
        elif name == "hybrid":
            return hybrid_strategy
        raise ValueError("Unknown")

    mock_factory.get_strategy.side_effect = get_strat

    service = ComparisonService(factory=mock_factory)
    request = ResearchComparisonRequest(
        query="GraphRAG architecture",
        strategies=["vector", "graph", "hybrid"],
        k=5,
    )

    response = asyncio.run(service.compare(request))

    assert response.query == "GraphRAG architecture"
    assert len(response.results) == 3
    assert response.results[0].strategy == "vector"
    assert response.results[1].strategy == "graph"
    assert response.results[2].strategy == "hybrid"

    # Hybrid retrieved 5 docs vs vector's 3 docs vs graph's 0 docs -> Hybrid is best
    assert response.best_strategy == "hybrid"
    assert "Selected 'hybrid' as best strategy" in response.reason
    assert "T" in response.timestamp  # ISO-8601 format check


def test_comparison_service_tie_breaker_lowest_latency():
    mock_factory = MagicMock()

    vector_strategy = AsyncMock()
    vector_strategy.run.return_value = StrategyResult(
        strategy="vector",
        latency_ms=20.0,  # lower latency
        documents_retrieved=5,
        graph_entities=0,
        citations=[],
        retrieved_items=[],
        success=True,
    )

    hybrid_strategy = AsyncMock()
    hybrid_strategy.run.return_value = StrategyResult(
        strategy="hybrid",
        latency_ms=50.0,  # higher latency
        documents_retrieved=5,  # tied on documents_retrieved
        graph_entities=2,
        citations=[],
        retrieved_items=[],
        success=True,
    )

    def get_strat(name):
        if name == "vector":
            return vector_strategy
        if name == "hybrid":
            return hybrid_strategy
        raise ValueError()

    mock_factory.get_strategy.side_effect = get_strat

    service = ComparisonService(factory=mock_factory)
    request = ResearchComparisonRequest(
        query="latency tie test",
        strategies=["vector", "hybrid"],
        k=5,
    )

    response = asyncio.run(service.compare(request))

    assert response.best_strategy == "vector"
    assert "lowest latency" in response.reason.lower()


def test_comparison_service_partial_failure_resilience():
    mock_factory = MagicMock()

    failing_vector_strategy = AsyncMock()
    failing_vector_strategy.run.side_effect = RuntimeError("Database connection failed")

    graph_strategy = AsyncMock()
    graph_strategy.run.return_value = StrategyResult(
        strategy="graph",
        latency_ms=15.0,
        documents_retrieved=0,
        graph_entities=2,
        citations=[],
        retrieved_items=[],
        success=True,
    )

    def get_strat(name):
        if name == "vector":
            return failing_vector_strategy
        elif name == "graph":
            return graph_strategy
        raise ValueError()

    mock_factory.get_strategy.side_effect = get_strat

    service = ComparisonService(factory=mock_factory)
    request = ResearchComparisonRequest(
        query="resilience test",
        strategies=["vector", "graph"],
        k=5,
    )

    response = asyncio.run(service.compare(request))

    assert len(response.results) == 2

    # Vector failed
    assert response.results[0].strategy == "vector"
    assert response.results[0].success is False
    assert "Database connection failed" in response.results[0].error

    # Graph succeeded
    assert response.results[1].strategy == "graph"
    assert response.results[1].success is True

    # Graph strategy selected as best among surviving strategies
    assert response.best_strategy == "graph"
