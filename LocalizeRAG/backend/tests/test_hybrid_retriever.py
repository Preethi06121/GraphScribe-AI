import asyncio
from unittest.mock import MagicMock

import pytest

from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.retrieval import (
    Citation,
    ContextDocument,
    ContextGraphItem,
    GraphHit,
    HybridContext,
    HybridContextMetadata,
    ProcessedQuery,
    RankedItem,
    VectorHit,
)


@pytest.fixture
def hybrid_components():
    query_processor = MagicMock()
    query_processor.process.return_value = ProcessedQuery(
        original="What is GraphRAG?",
        normalized="what is graphrag",
        keywords=["graphrag"],
        semantic_query="what is graphrag",
    )

    vector_retriever = MagicMock()
    vector_retriever.retrieve.return_value = [
        VectorHit(
            chunk="GraphRAG uses knowledge graphs",
            score=0.95,
            page=1,
            document_id="doc-1",
            document_name="paper.pdf",
            chunk_id="c1",
        )
    ]

    graph_retriever = MagicMock()
    graph_retriever.retrieve.return_value = [
        GraphHit(
            entity_name="GraphRAG",
            entity_type="TECH_TERM",
            document_id="doc-1",
            document_name="paper.pdf",
            page_number=1,
            score=0.8,
        )
    ]

    ranking_service = MagicMock()
    ranking_service.vector_weight = 0.7
    ranking_service.graph_weight = 0.3
    ranking_service.rank.return_value = [
        RankedItem(
            item_id="c1",
            item_type="document",
            content="GraphRAG uses knowledge graphs",
            document_id="doc-1",
            document_name="paper.pdf",
            page=1,
            vector_score=0.95,
            graph_score=0.8,
            final_score=0.905,
            source="vector",
        )
    ]

    context_fusion = MagicMock()
    context_fusion.fuse.return_value = HybridContext(
        documents=[
            ContextDocument(
                chunk="GraphRAG uses knowledge graphs",
                score=0.905,
                page=1,
                document_id="doc-1",
                document_name="paper.pdf",
                chunk_id="c1",
            )
        ],
        graph=[
            ContextGraphItem(
                entity_name="GraphRAG",
                entity_type="TECH_TERM",
                document_id="doc-1",
                document_name="paper.pdf",
                page_number=1,
                score=0.8,
            )
        ],
        citations=[
            Citation(
                document_id="doc-1",
                document_name="paper.pdf",
                page=1,
                source="vector",
                reference="c1",
            )
        ],
        metadata=HybridContextMetadata(
            query="What is GraphRAG?",
            normalized_query="what is graphrag",
            keywords=["graphrag"],
            vector_hits=1,
            graph_hits=1,
            vector_weight=0.7,
            graph_weight=0.3,
        ),
    )

    retriever = HybridRetriever(
        query_processor=query_processor,
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        ranking_service=ranking_service,
        context_fusion=context_fusion,
        cache_size=8,
        top_k=5,
    )
    return retriever, query_processor, vector_retriever, graph_retriever


def test_hybrid_retrieve_pipeline(hybrid_components):
    retriever, query_processor, vector_retriever, graph_retriever = hybrid_components

    context = asyncio.run(retriever.retrieve("What is GraphRAG?"))

    assert len(context.documents) == 1
    assert context.documents[0].document_id == "doc-1"
    query_processor.process.assert_called_once()
    vector_retriever.retrieve.assert_called_once()
    graph_retriever.retrieve.assert_called_once()


def test_hybrid_retrieve_uses_cache(hybrid_components):
    retriever, query_processor, vector_retriever, graph_retriever = hybrid_components

    first = asyncio.run(retriever.retrieve("What is GraphRAG?"))
    second = asyncio.run(retriever.retrieve("  what is GraphRAG?  "))

    assert first == second
    assert query_processor.process.call_count == 1
    assert vector_retriever.retrieve.call_count == 1
    assert graph_retriever.retrieve.call_count == 1
