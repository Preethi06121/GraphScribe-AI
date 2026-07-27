import logging
from pathlib import Path

import fitz

from app.schemas.document import PageContent

logger = logging.getLogger(__name__)


class PDFDocumentLoader:
    """Extracts text from PDF files using PyMuPDF."""

    def load(self, file_path: str | Path) -> list[PageContent]:
        path = Path(file_path)
        logger.info("Extracting text from PDF: %s", path.name)

        pages: list[PageContent] = []

        try:
            with fitz.open(path) as document:
                for page_index in range(len(document)):
                    page = document[page_index]
                    text = page.get_text().strip()
                    if text:
                        pages.append(
                            PageContent(
                                page_number=page_index + 1,
                                text=text,
                            )
                        )
        except fitz.FileDataError as exc:
            logger.exception("Failed to read PDF file: %s", path.name)
            raise ValueError(f"Invalid or corrupted PDF file: {path.name}") from exc
        except Exception as exc:
            logger.exception("Unexpected error loading PDF: %s", path.name)
            raise ValueError(f"Failed to extract text from PDF: {path.name}") from exc

        if not pages:
            logger.warning("No extractable text found in PDF: %s", path.name)
            raise ValueError(f"No extractable text found in PDF: {path.name}")

        logger.info("Extracted text from %d page(s) in %s", len(pages), path.name)
        return pages
