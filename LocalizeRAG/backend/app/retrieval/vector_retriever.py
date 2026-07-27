import logging

from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import VectorStore
from app.schemas.retrieval import ProcessedQuery, VectorHit

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieves top-k document chunks from ChromaDB."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._top_k = top_k

    def retrieve(self, processed_query: ProcessedQuery, top_k: int | None = None) -> list[VectorHit]:
        k = top_k if top_k is not None else self._top_k
        logger.info("Running vector retrieval for top-%d", k)

        embeddings = self._embedding_service.embed_texts([processed_query.semantic_query])
        raw_hits = self._vector_store.query(embeddings[0], top_k=k)

        hits = [
            VectorHit(
                chunk=hit["chunk"],
                score=float(hit["score"]),
                page=int(hit["page"]),
                document_id=str(hit["document_id"]),
                document_name=str(hit.get("document_name", "")),
                chunk_id=str(hit.get("chunk_id", "")),
                source=str(hit.get("source", "")),
            )
            for hit in raw_hits
        ]

        logger.info("Vector retriever returned %d hit(s)", len(hits))
        return hits
