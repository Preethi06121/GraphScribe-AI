from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_hybrid_retriever
from app.main import app
from app.schemas.retrieval import (
    Citation,
    ContextDocument,
    ContextGraphItem,
    HybridContext,
    HybridContextMetadata,
)


@pytest.fixture
def retrieval_client() -> TestClient:
    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = HybridContext(
        documents=[
            ContextDocument(
                chunk="Hybrid retrieval combines vector and graph search",
                score=0.88,
                page=1,
                document_id="doc-1",
                document_name="guide.pdf",
                chunk_id="c1",
            )
        ],
        graph=[
            ContextGraphItem(
                entity_name="GraphRAG",
                entity_type="TECH_TERM",
                document_id="doc-1",
                document_name="guide.pdf",
                page_number=1,
                score=0.7,
            )
        ],
        citations=[
            Citation(
                document_id="doc-1",
                document_name="guide.pdf",
                page=1,
                source="vector",
                reference="c1",
            )
        ],
        metadata=HybridContextMetadata(
            query="How does hybrid retrieval work?",
            normalized_query="how does hybrid retrieval work",
            keywords=["hybrid", "retrieval", "work"],
            vector_hits=1,
            graph_hits=1,
            vector_weight=0.7,
            graph_weight=0.3,
        ),
    )
    app.dependency_overrides[get_hybrid_retriever] = lambda: mock_retriever
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_retrieval_search_endpoint(retrieval_client: TestClient):
    response = retrieval_client.post(
        "/retrieval/search",
        json={"query": "How does hybrid retrieval work?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "graph" in data
    assert "citations" in data
    assert "metadata" in data
    assert data["documents"][0]["document_id"] == "doc-1"
    assert data["metadata"]["vector_weight"] == 0.7


def test_retrieval_search_rejects_empty_query(retrieval_client: TestClient):
    response = retrieval_client.post(
        "/retrieval/search",
        json={"query": "   "},
    )

    assert response.status_code == 400
