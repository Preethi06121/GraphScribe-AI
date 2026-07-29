from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.core.deps import get_comparison_service
from app.main import app
from app.research.schemas import (
    ResearchComparisonResponse,
    StrategyResult,
)


def test_research_compare_api_endpoint():
    mock_service = AsyncMock()
    mock_service.compare.return_value = ResearchComparisonResponse(
        query="How does GraphRAG work?",
        results=[
            StrategyResult(
                strategy="vector",
                latency_ms=25.0,
                documents_retrieved=3,
                graph_entities=0,
                citations=["doc1.pdf (Page 1)"],
                retrieved_items=[],
                success=True,
            ),
            StrategyResult(
                strategy="graph",
                latency_ms=18.0,
                documents_retrieved=0,
                graph_entities=4,
                citations=["GraphRAG [USES] Neo4j"],
                retrieved_items=[],
                success=True,
            ),
            StrategyResult(
                strategy="hybrid",
                latency_ms=35.0,
                documents_retrieved=4,
                graph_entities=4,
                citations=["doc1.pdf (Page 1)"],
                retrieved_items=[],
                success=True,
            ),
        ],
        best_strategy="hybrid",
        reason="Selected 'hybrid' as best strategy because it retrieved 4 document(s).",
        timestamp="2026-07-29T11:00:00+00:00",
    )

    app.dependency_overrides[get_comparison_service] = lambda: mock_service

    client = TestClient(app)
    payload = {"query": "How does GraphRAG work?"}
    response = client.post("/research/compare", json=payload)

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "How does GraphRAG work?"
    assert len(data["results"]) == 3
    assert data["best_strategy"] == "hybrid"
    assert "timestamp" in data


def test_research_compare_api_empty_query_400():
    client = TestClient(app)
    response = client.post("/research/compare", json={"query": "   "})
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]
