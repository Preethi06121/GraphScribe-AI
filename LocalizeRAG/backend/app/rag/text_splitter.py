import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas.document import PageContent, TextChunk

logger = logging.getLogger(__name__)


class RecursiveTextSplitter:
    """Splits page content into chunks using LangChain's RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_pages(
        self,
        pages: list[PageContent],
        document_id: str,
        document_name: str,
        source: str,
    ) -> list[TextChunk]:
        logger.info("Chunking %d page(s) for document: %s", len(pages), document_name)

        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in pages:
            page_chunks = self._splitter.split_text(page.text)
            for content in page_chunks:
                chunk_id = f"{document_name}_chunk_{chunk_index}"
                chunks.append(
                    TextChunk(
                        content=content,
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page.page_number,
                        chunk_id=chunk_id,
                        source=source,
                    )
                )
                chunk_index += 1

        if not chunks:
            logger.warning("No chunks produced for document: %s", document_name)
            raise ValueError(f"No chunks produced for document: {document_name}")

        logger.info("Produced %d chunk(s) for document: %s", len(chunks), document_name)
        return chunks
