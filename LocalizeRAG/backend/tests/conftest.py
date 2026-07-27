from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.document import IngestionResult


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "LocalizeRAG test content. " * 50)
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "multi_page.pdf"
    document = fitz.open()
    for page_number in range(3):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"Page {page_number + 1} content. " * 200,
        )
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture
def mock_ingestion_result() -> IngestionResult:
    return IngestionResult(
        document_id="550e8400-e29b-41d4-a716-446655440000",
        document_name="sample.pdf",
        pages=1,
        chunks=3,
    )
