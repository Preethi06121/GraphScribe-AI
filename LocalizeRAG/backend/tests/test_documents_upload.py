from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_graph_builder, get_ingestion_pipeline
from app.main import app
from app.schemas.document import IngestionResult


@pytest.fixture
def upload_client(mock_ingestion_result: IngestionResult) -> TestClient:
    mock_pipeline = AsyncMock()
    mock_pipeline.ingest.return_value = mock_ingestion_result

    mock_graph_builder = AsyncMock()
    mock_graph_builder.build_from_file.return_value = None

    app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_graph_builder] = lambda: mock_graph_builder
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_upload_pdf_success(upload_client: TestClient, sample_pdf):
    with sample_pdf.open("rb") as pdf_file:
        response = upload_client.post(
            "/documents/upload",
            files={"file": ("sample.pdf", pdf_file, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["document_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert data["document"] == "sample.pdf"
    assert data["pages"] == 1
    assert data["chunks"] == 3


def test_upload_rejects_non_pdf(upload_client: TestClient):
    response = upload_client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are allowed."


def test_upload_rejects_empty_file(upload_client: TestClient):
    response = upload_client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."
