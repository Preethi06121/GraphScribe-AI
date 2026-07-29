import asyncio
from unittest.mock import MagicMock

from app.research.strategies.graph_strategy import GraphStrategy
from app.research.strategies.hybrid_strategy import HybridStrategy
from app.research.strategies.vector_strategy import VectorStrategy
from app.schemas.retrieval import (
    Citation,
    ContextDocument,
    ContextGraphItem,
    GraphHit,
    HybridContext,
    HybridContextMetadata,
    VectorHit,
)


def test_vector_strategy_run():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        VectorHit(
            chunk="GraphRAG integrates knowledge graphs with vector embeddings.",
            score=0.89,
            page=1,
            document_id="doc-123",
            document_name="graphrag_paper.pdf",
            chunk_id="chunk-1",
            source="pdf",
        )
    ]

    strategy = VectorStrategy(vector_retriever=mock_retriever)
    result = asyncio.run(
        strategy.run("How does GraphRAG work?", k=5, include_items=True)
    )

    assert result.strategy == "vector"
    assert result.success is True
    assert result.documents_retrieved == 1
    assert result.graph_entities == 0
    assert len(result.citations) == 1
    assert "graphrag_paper.pdf" in result.citations[0]
    assert len(result.retrieved_items) == 1
    assert result.retrieved_items[0]["chunk_id"] == "chunk-1"


def test_graph_strategy_run():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        GraphHit(
            entity_name="GraphRAG",
            entity_type="CONCEPT",
            connected_entity="KnowledgeGraph",
            connected_type="SYSTEM",
            relationship_type="USES",
            document_id="doc-123",
            document_name="graphrag_paper.pdf",
            page_number=2,
            score=0.95,
        )
    ]

    strategy = GraphStrategy(graph_retriever=mock_retriever)
    result = asyncio.run(
        strategy.run("Tell me about GraphRAG", k=5, include_items=True)
    )

    assert result.strategy == "graph"
    assert result.success is True
    assert result.documents_retrieved == 0
    assert result.graph_entities == 1
    assert len(result.citations) == 1
    assert "GraphRAG [USES] KnowledgeGraph" in result.citations[0]
    assert len(result.retrieved_items) == 1
    assert result.retrieved_items[0]["entity_name"] == "GraphRAG"


def test_hybrid_strategy_run():
    mock_retriever = MagicMock()

    doc_hit = ContextDocument(
        chunk="Vector chunk text",
        score=0.8,
        page=1,
        document_id="doc-123",
        document_name="doc.pdf",
        chunk_id="c1",
        source="vector",
    )
    graph_hit = ContextGraphItem(
        entity_name="GraphRAG",
        entity_type="CONCEPT",
        connected_entity=None,
        relationship_type=None,
        document_id="doc-123",
        document_name="doc.pdf",
        page_number=1,
        score=0.85,
        source="graph",
    )

    context = HybridContext(
        documents=[doc_hit],
        graph=[graph_hit],
        citations=[
            Citation(
                document_id="doc-123",
                document_name="doc.pdf",
                page=1,
                source="vector",
                reference="c1",
            )
        ],
        metadata=HybridContextMetadata(
            query="hybrid test",
            normalized_query="hybrid test",
            keywords=["hybrid"],
            vector_hits=1,
            graph_hits=1,
            vector_weight=0.7,
            graph_weight=0.3,
        ),
    )

    async def async_retrieve(query):
        return context

    mock_retriever.retrieve = MagicMock(side_effect=async_retrieve)

    strategy = HybridStrategy(hybrid_retriever=mock_retriever)
    result = asyncio.run(strategy.run("hybrid query", k=5, include_items=True))

    assert result.strategy == "hybrid"
    assert result.success is True
    assert result.documents_retrieved == 1
    assert result.graph_entities == 1
    assert len(result.citations) == 1
    assert len(result.retrieved_items) == 2
