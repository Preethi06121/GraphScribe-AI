import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.core.deps import get_generation_engine
from app.llm.explainability import ExplainabilityGenerator
from app.llm.generation_engine import GenerationEngine
from app.llm.prompt_builder import PromptBuilder
from app.llm.response_formatter import ResponseFormatter
from app.main import app
from app.research.factory import StrategyFactory
from app.schemas.retrieval import (
    Citation,
    ContextDocument,
    ContextGraphItem,
    GraphHit,
    HybridContext,
    HybridContextMetadata,
    VectorHit,
)


def _build_mock_generation_engine():
    mock_vector_retriever = MagicMock()
    mock_vector_retriever.retrieve.return_value = [
        VectorHit(
            chunk="Vector hit chunk for strategy aware generation.",
            score=0.9,
            page=1,
            document_id="doc-vec",
            document_name="vector_doc.pdf",
            chunk_id="chunk-vec",
        )
    ]

    mock_graph_retriever = MagicMock()
    mock_graph_retriever.retrieve.return_value = [
        GraphHit(
            entity_name="GraphRAG",
            entity_type="CONCEPT",
            connected_entity="KnowledgeGraph",
            connected_type="SYSTEM",
            relationship_type="USES",
            document_id="doc-graph",
            document_name="graph_doc.pdf",
            page_number=2,
            score=0.85,
        )
    ]

    mock_hybrid_retriever = AsyncMock()
    mock_hybrid_retriever.retrieve.return_value = HybridContext(
        documents=[
            ContextDocument(
                chunk="Hybrid chunk text",
                score=0.92,
                page=1,
                document_id="doc-hybrid",
                document_name="hybrid_doc.pdf",
                chunk_id="c-hy",
            )
        ],
        graph=[
            ContextGraphItem(
                entity_name="GraphRAG",
                entity_type="CONCEPT",
                document_id="doc-hybrid",
                document_name="hybrid_doc.pdf",
                page_number=1,
                score=0.88,
            )
        ],
        citations=[
            Citation(
                document_id="doc-hybrid",
                document_name="hybrid_doc.pdf",
                page=1,
                source="hybrid",
                reference="c-hy",
            )
        ],
        metadata=HybridContextMetadata(
            query="test query",
            normalized_query="test query",
            keywords=["test"],
            vector_hits=1,
            graph_hits=1,
            vector_weight=0.7,
            graph_weight=0.3,
        ),
    )

    query_processor = MagicMock()
    query_processor.process.return_value = MagicMock(
        normalized="test query", keywords=["test"]
    )

    factory = StrategyFactory(
        vector_retriever=mock_vector_retriever,
        graph_retriever=mock_graph_retriever,
        hybrid_retriever=mock_hybrid_retriever,
        query_processor=query_processor,
    )

    provider = AsyncMock()
    provider.generate.return_value = json.dumps(
        {
            "title": "Strategy Aware Article",
            "summary": "Article generated with specific retrieval strategy.",
            "sections": [
                {
                    "heading": "Section 1",
                    "content": "Content grounded by selected strategy.",
                }
            ],
            "conclusion": "Strategy aware generation succeeded.",
        }
    )

    engine = GenerationEngine(
        strategy_factory=factory,
        prompt_builder=PromptBuilder(),
        provider=provider,
        response_formatter=ResponseFormatter(),
        explainability_generator=ExplainabilityGenerator(),
        hybrid_retriever=mock_hybrid_retriever,
    )
    return engine, mock_vector_retriever, mock_graph_retriever, mock_hybrid_retriever


def test_generation_engine_vector_strategy():
    engine, vector_retriever, graph_retriever, hybrid_retriever = (
        _build_mock_generation_engine()
    )

    article = asyncio.run(
        engine.generate_article(
            topic="GraphRAG",
            audience="Engineers",
            country="India",
            tone="Professional",
            length=1500,
            retrieval_strategy="VECTOR",
        )
    )

    assert article.title == "Strategy Aware Article"
    assert article.explainability.chunk_count == 1
    assert article.explainability.entity_count == 0
    vector_retriever.retrieve.assert_called_once()


def test_generation_engine_graph_strategy():
    engine, vector_retriever, graph_retriever, hybrid_retriever = (
        _build_mock_generation_engine()
    )

    article = asyncio.run(
        engine.generate_article(
            topic="GraphRAG",
            audience="Engineers",
            country="India",
            tone="Professional",
            length=1500,
            retrieval_strategy="GRAPH",
        )
    )

    assert article.title == "Strategy Aware Article"
    assert article.explainability.chunk_count == 0
    assert article.explainability.entity_count == 1
    graph_retriever.retrieve.assert_called_once()


def test_generation_engine_hybrid_strategy_default():
    engine, vector_retriever, graph_retriever, hybrid_retriever = (
        _build_mock_generation_engine()
    )

    article = asyncio.run(
        engine.generate_article(
            topic="GraphRAG",
            audience="Engineers",
            country="India",
            tone="Professional",
            length=1500,
            # Omitting retrieval_strategy -> defaults to HYBRID
        )
    )

    assert article.title == "Strategy Aware Article"
    assert article.explainability.chunk_count == 1
    assert article.explainability.entity_count == 1
    hybrid_retriever.retrieve.assert_awaited_once()


def test_generation_engine_backward_compatibility_hybrid_retriever_init():
    mock_hybrid = AsyncMock()
    mock_hybrid.retrieve.return_value = HybridContext(
        documents=[],
        graph=[],
        citations=[],
        metadata=HybridContextMetadata(
            query="test",
            normalized_query="test",
            keywords=[],
            vector_hits=0,
            graph_hits=0,
            vector_weight=0.7,
            graph_weight=0.3,
        ),
    )
    provider = AsyncMock()
    provider.generate.return_value = json.dumps(
        {
            "title": "Legacy Article",
            "summary": "Summary",
            "sections": [],
            "conclusion": "Conclusion",
        }
    )

    legacy_engine = GenerationEngine(
        hybrid_retriever=mock_hybrid,
        prompt_builder=PromptBuilder(),
        provider=provider,
        response_formatter=ResponseFormatter(),
        explainability_generator=ExplainabilityGenerator(),
    )

    article = asyncio.run(
        legacy_engine.generate_article(
            topic="GraphRAG",
            audience="Students",
            country="India",
            tone="Professional",
            length=1500,
        )
    )

    assert article.title == "Legacy Article"
    mock_hybrid.retrieve.assert_awaited_once()


def test_content_api_with_graph_strategy():
    engine, _, _, _ = _build_mock_generation_engine()
    app.dependency_overrides[get_generation_engine] = lambda: engine

    client = TestClient(app)
    response = client.post(
        "/content/generate",
        json={
            "topic": "GraphRAG",
            "audience": "Engineering Students",
            "country": "India",
            "tone": "Professional",
            "length": 1500,
            "retrieval_strategy": "GRAPH",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Strategy Aware Article"


def test_content_api_invalid_strategy_400():
    client = TestClient(app)
    response = client.post(
        "/content/generate",
        json={
            "topic": "GraphRAG",
            "audience": "Engineering Students",
            "country": "India",
            "tone": "Professional",
            "length": 1500,
            "retrieval_strategy": "INVALID_STRATEGY_NAME",
        },
    )

    assert response.status_code == 400
    assert "Invalid retrieval strategy" in response.json()["detail"]
