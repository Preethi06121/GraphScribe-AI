import logging
from pathlib import Path

import chromadb

from app.schemas.document import TextChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Persists document chunks and embeddings in ChromaDB."""

    def __init__(self, persist_directory: str, collection_name: str) -> None:
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection ready: %s at %s",
            collection_name,
            self._persist_directory,
        )

    def add_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        logger.info("Storing %d chunk(s) in ChromaDB", len(chunks))

        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                }
                for chunk in chunks
            ],
        )

        logger.info("Successfully stored %d chunk(s) in ChromaDB", len(chunks))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve the top-k most similar chunks for a query embedding."""
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        logger.info("Querying ChromaDB for top-%d chunks", top_k)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict] = []
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]

        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else 1.0
            score = max(0.0, 1.0 - float(distance))
            hits.append(
                {
                    "chunk": document or "",
                    "score": score,
                    "page": int(metadata.get("page_number", 0)),
                    "document_id": str(metadata.get("document_id", "")),
                    "document_name": str(metadata.get("document_name", "")),
                    "chunk_id": str(metadata.get("chunk_id", ids[index] if index < len(ids) else "")),
                    "source": str(metadata.get("source", "")),
                }
            )

        logger.info("ChromaDB returned %d hit(s)", len(hits))
        return hits
