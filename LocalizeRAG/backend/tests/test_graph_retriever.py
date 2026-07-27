from unittest.mock import MagicMock

from app.retrieval.graph_retriever import GraphRetriever
from app.schemas.graph import GraphEntity
from app.schemas.retrieval import ProcessedQuery


def test_graph_retriever_scores_connected_entities():
    neo4j = MagicMock()
    neo4j.is_available.return_value = True
    neo4j.search_by_entity_names.return_value = [
        {
            "entity_name": "OpenAI",
            "entity_type": "ORG",
            "document_id": "doc-1",
            "document_name": "ai.pdf",
            "page_number": 1,
            "connected_entity": "GPT-4",
            "connected_type": "PRODUCT",
            "relationship_type": "DEVELOPED",
        }
    ]

    entity_extractor = MagicMock()
    entity_extractor.extract_from_text.return_value = [
        GraphEntity(
            entity_name="OpenAI",
            entity_type="ORG",
            document_id="query",
            document_name="query",
            page_number=0,
        )
    ]

    retriever = GraphRetriever(neo4j, entity_extractor, top_k=5)
    processed = ProcessedQuery(
        original="OpenAI developed GPT-4",
        normalized="openai developed gpt-4",
        keywords=["openai", "developed", "gpt-4"],
        semantic_query="openai developed gpt-4",
    )

    hits = retriever.retrieve(processed)

    assert len(hits) == 1
    assert hits[0].entity_name == "OpenAI"
    assert hits[0].connected_entity == "GPT-4"
    assert hits[0].relationship_type == "DEVELOPED"
    assert hits[0].document_id == "doc-1"
    assert hits[0].score > 0.5


def test_graph_retriever_handles_unavailable_neo4j():
    neo4j = MagicMock()
    neo4j.is_available.return_value = False
    entity_extractor = MagicMock()
    entity_extractor.extract_from_text.return_value = []

    retriever = GraphRetriever(neo4j, entity_extractor)
    processed = ProcessedQuery(
        original="RAG embeddings",
        normalized="rag embeddings",
        keywords=["rag", "embeddings"],
        semantic_query="rag embeddings",
    )

    hits = retriever.retrieve(processed)

    assert hits == []
    neo4j.search_by_entity_names.assert_not_called()
