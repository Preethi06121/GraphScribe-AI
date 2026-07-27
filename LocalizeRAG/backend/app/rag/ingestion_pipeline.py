import asyncio
import logging
from pathlib import Path

from app.rag.document_loader import PDFDocumentLoader
from app.rag.embedding_service import EmbeddingService
from app.rag.text_splitter import RecursiveTextSplitter
from app.rag.vector_store import VectorStore
from app.schemas.document import IngestionResult

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates PDF ingestion: extract, chunk, embed, and store."""

    def __init__(
        self,
        document_loader: PDFDocumentLoader,
        text_splitter: RecursiveTextSplitter,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._document_loader = document_loader
        self._text_splitter = text_splitter
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    async def ingest(
        self,
        file_path: str | Path,
        document_id: str,
        document_name: str,
    ) -> IngestionResult:
        path = Path(file_path)
        source = document_name

        logger.info(
            "Starting ingestion for document: %s (id=%s)",
            document_name,
            document_id,
        )

        pages = await asyncio.to_thread(self._document_loader.load, path)
        chunks = await asyncio.to_thread(
            self._text_splitter.split_pages,
            pages,
            document_id,
            document_name,
            source,
        )
        texts = [chunk.content for chunk in chunks]
        embeddings = await asyncio.to_thread(self._embedding_service.embed_texts, texts)
        await asyncio.to_thread(self._vector_store.add_chunks, chunks, embeddings)

        result = IngestionResult(
            document_id=document_id,
            document_name=document_name,
            pages=len(pages),
            chunks=len(chunks),
        )
        logger.info(
            "Ingestion complete for %s (id=%s): %d page(s), %d chunk(s)",
            document_name,
            document_id,
            result.pages,
            result.chunks,
        )
        return result
