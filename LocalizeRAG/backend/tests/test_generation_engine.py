import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.llm.generation_engine import GenerationEngine
from app.llm.prompt_builder import PromptBuilder
from app.llm.response_formatter import ResponseFormatter
from app.llm.explainability import ExplainabilityGenerator
from app.schemas.retrieval import (
    ContextDocument,
    ContextGraphItem,
    Citation,
    HybridContext,
    HybridContextMetadata,
)


@pytest.fixture
def sample_context() -> HybridContext:
    return HybridContext(
        documents=[
            ContextDocument(
                chunk="GraphRAG merges retrieval and graph reasoning.",
                score=0.9,
                page=1,
                document_id="doc-1",
                document_name="graphrag.pdf",
                chunk_id="c1",
            )
        ],
        graph=[
            ContextGraphItem(
                entity_name="GraphRAG",
                entity_type="TECH_TERM",
                document_id="doc-1",
                document_name="graphrag.pdf",
                page_number=1,
                score=0.8,
            )
        ],
        citations=[
            Citation(
                document_id="doc-1",
                document_name="graphrag.pdf",
                page=1,
                source="vector",
                reference="c1",
            )
        ],
        metadata=HybridContextMetadata(
            query="graphrag for engineering students in india",
            normalized_query="graphrag for engineering students in india",
            keywords=["graphrag", "engineering", "students", "india"],
            vector_hits=1,
            graph_hits=1,
            vector_weight=0.7,
            graph_weight=0.3,
        ),
    )


@pytest.fixture
def generation_engine(sample_context: HybridContext) -> GenerationEngine:
    hybrid_retriever = AsyncMock()
    hybrid_retriever.retrieve.return_value = sample_context

    provider = AsyncMock()
    provider.generate.return_value = json.dumps(
        {
            "title": "GraphRAG for Engineering Students",
            "summary": "An overview of GraphRAG for Indian engineering students.",
            "sections": [
                {
                    "heading": "Introduction",
                    "content": "GraphRAG combines vector retrieval with graph knowledge.",
                }
            ],
            "conclusion": "GraphRAG is a promising research direction.",
        }
    )

    return GenerationEngine(
        hybrid_retriever=hybrid_retriever,
        prompt_builder=PromptBuilder(),
        provider=provider,
        response_formatter=ResponseFormatter(),
        explainability_generator=ExplainabilityGenerator(),
    )


def test_generation_engine_pipeline(generation_engine: GenerationEngine):
    article = asyncio.run(
        generation_engine.generate_article(
            topic="GraphRAG",
            audience="Engineering Students",
            country="India",
            tone="Professional",
            length=1500,
        )
    )

    assert article.title == "GraphRAG for Engineering Students"
    assert article.metadata.country == "India"
    assert article.metadata.retrieval_strategy == "HybridRAG"
    assert article.explainability.chunk_count == 1
    assert article.explainability.entity_count == 1
    assert len(article.sections) == 1


def test_generation_engine_calls_retriever_and_provider(generation_engine: GenerationEngine):
    asyncio.run(
        generation_engine.generate_article(
            topic="GraphRAG",
            audience="Engineering Students",
            country="India",
            tone="Professional",
            length=1500,
        )
    )

    generation_engine._hybrid_retriever.retrieve.assert_awaited_once()
    generation_engine._provider.generate.assert_awaited_once()
