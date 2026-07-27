from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_generation_engine
from app.main import app
from app.schemas.content import (
    ArticleMetadata,
    ArticleSection,
    Explainability,
    GeneratedArticleResponse,
)


@pytest.fixture
def content_client() -> TestClient:
    mock_engine = AsyncMock()
    mock_engine.generate_article.return_value = GeneratedArticleResponse(
        title="GraphRAG for Engineering Students",
        summary="A localized overview of GraphRAG.",
        sections=[
            ArticleSection(
                heading="Introduction",
                content="GraphRAG combines vector and graph retrieval.",
            )
        ],
        conclusion="GraphRAG enables richer grounded generation.",
        citations=[],
        metadata=ArticleMetadata(
            retrieval_strategy="HybridRAG",
            country="India",
            audience="Engineering Students",
            tone="Professional",
            topic="GraphRAG",
            word_count=120,
            target_length=1500,
        ),
        explainability=Explainability(
            vector_chunks=[],
            graph_entities=[],
            reasoning="Hybrid retrieval selected relevant chunks and entities.",
            documents_used=[],
            retrieval_strategy="HybridRAG",
            vector_score_summary=0.0,
            graph_score_summary=0.0,
            chunk_count=0,
            entity_count=0,
        ),
    )

    app.dependency_overrides[get_generation_engine] = lambda: mock_engine
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_content_generate_endpoint(content_client: TestClient):
    response = content_client.post(
        "/content/generate",
        json={
            "topic": "GraphRAG",
            "audience": "Engineering Students",
            "country": "India",
            "tone": "Professional",
            "length": 1500,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "GraphRAG for Engineering Students"
    assert data["metadata"]["country"] == "India"
    assert "explainability" in data
    assert "sections" in data
