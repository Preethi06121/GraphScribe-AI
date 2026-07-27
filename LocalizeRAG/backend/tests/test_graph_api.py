from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_neo4j_service
from app.main import app
from app.schemas.graph import (
    DocumentGraphNode,
    DocumentGraphRelationship,
    DocumentGraphResponse,
    DocumentGraphStatistics,
    GraphStatistics,
)


@pytest.fixture
def graph_client() -> TestClient:
    mock_service = MagicMock()
    app.dependency_overrides[get_neo4j_service] = lambda: mock_service
    client = TestClient(app)
    yield client, mock_service
    app.dependency_overrides.clear()


def test_graph_statistics_endpoint(graph_client):
    client, mock_service = graph_client
    mock_service.is_available.return_value = True
    mock_service.get_statistics.return_value = GraphStatistics(
        nodes=25,
        relationships=12,
        entity_types=["ORG", "PERSON", "TECH_TERM"],
        documents=3,
    )

    response = client.get("/graph/statistics")

    assert response.status_code == 200
    assert response.json() == {
        "nodes": 25,
        "relationships": 12,
        "entity_types": ["ORG", "PERSON", "TECH_TERM"],
        "documents": 3,
    }


def test_graph_statistics_unavailable(graph_client):
    client, mock_service = graph_client
    mock_service.is_available.return_value = False

    response = client.get("/graph/statistics")

    assert response.status_code == 503


def test_graph_document_endpoint(graph_client):
    client, mock_service = graph_client
    document_id = "550e8400-e29b-41d4-a716-446655440000"
    mock_service.is_available.return_value = True
    mock_service.get_document_graph.return_value = DocumentGraphResponse(
        document_id=document_id,
        nodes=[
            DocumentGraphNode(
                entity_name="OpenAI",
                entity_type="ORG",
                document_id=document_id,
                document_name="research.pdf",
                page_number=1,
            )
        ],
        relationships=[
            DocumentGraphRelationship(
                source_entity="OpenAI",
                source_type="ORG",
                relationship_type="DEVELOPED",
                target_entity="GPT-4",
                target_type="PRODUCT",
                document_id=document_id,
                source_document="research.pdf",
                page_number=1,
            )
        ],
        statistics=DocumentGraphStatistics(
            nodes=1,
            relationships=1,
            entity_types=["ORG"],
        ),
    )

    response = client.get(f"/graph/document/{document_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == document_id
    assert len(data["nodes"]) == 1
    assert len(data["relationships"]) == 1
    assert data["statistics"]["nodes"] == 1


def test_graph_document_not_found(graph_client):
    client, mock_service = graph_client
    document_id = "00000000-0000-0000-0000-000000000000"
    mock_service.is_available.return_value = True
    mock_service.get_document_graph.return_value = DocumentGraphResponse(
        document_id=document_id,
        nodes=[],
        relationships=[],
        statistics=DocumentGraphStatistics(
            nodes=0,
            relationships=0,
            entity_types=[],
        ),
    )

    response = client.get(f"/graph/document/{document_id}")

    assert response.status_code == 404
