import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.deps import get_graph_builder, get_ingestion_pipeline
from app.main import app
from app.schemas.document import IngestionResult


def test_document_id_is_uuid4():
    generated = str(uuid.uuid4())
    parsed = uuid.UUID(generated, version=4)
    assert str(parsed) == generated


def test_upload_response_includes_document_id(sample_pdf, mock_ingestion_result: IngestionResult):
    mock_pipeline = AsyncMock()
    mock_pipeline.ingest.return_value = mock_ingestion_result
    mock_graph_builder = AsyncMock()

    app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_graph_builder] = lambda: mock_graph_builder

    try:
        client = TestClient(app)
        with sample_pdf.open("rb") as pdf_file:
            response = client.post(
                "/documents/upload",
                files={"file": ("sample.pdf", pdf_file, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        uuid.UUID(data["document_id"], version=4)
    finally:
        app.dependency_overrides.clear()
