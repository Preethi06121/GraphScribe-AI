from unittest.mock import MagicMock

from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.retrieval import ProcessedQuery


def test_vector_retriever_returns_hits():
    vector_store = MagicMock()
    embedding_service = MagicMock()
    embedding_service.embed_texts.return_value = [[0.1, 0.2, 0.3]]
    vector_store.query.return_value = [
        {
            "chunk": "Hybrid GraphRAG content",
            "score": 0.91,
            "page": 2,
            "document_id": "doc-1",
            "document_name": "paper.pdf",
            "chunk_id": "paper.pdf_chunk_0",
            "source": "paper.pdf",
        }
    ]

    retriever = VectorRetriever(vector_store, embedding_service, top_k=3)
    processed = ProcessedQuery(
        original="What is GraphRAG?",
        normalized="what is graphrag",
        keywords=["graphrag"],
        semantic_query="what is graphrag",
    )

    hits = retriever.retrieve(processed)

    assert len(hits) == 1
    assert hits[0].chunk == "Hybrid GraphRAG content"
    assert hits[0].score == 0.91
    assert hits[0].page == 2
    assert hits[0].document_id == "doc-1"
    embedding_service.embed_texts.assert_called_once_with(["what is graphrag"])
    vector_store.query.assert_called_once()
