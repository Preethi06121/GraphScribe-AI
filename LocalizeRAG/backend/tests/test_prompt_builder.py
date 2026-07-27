import pytest

from app.llm.prompt_builder import PromptBuilder, SUPPORTED_COUNTRIES
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
                chunk="GraphRAG combines vector retrieval with knowledge graphs.",
                score=0.92,
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
                connected_entity="Knowledge Graph",
                relationship_type="RELATED_TO",
                document_id="doc-1",
                document_name="graphrag.pdf",
                page_number=1,
                score=0.81,
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
            query="graphrag",
            normalized_query="graphrag",
            keywords=["graphrag"],
            vector_hits=1,
            graph_hits=1,
            vector_weight=0.7,
            graph_weight=0.3,
        ),
    )


def test_prompt_builder_includes_all_sections(sample_context: HybridContext):
    builder = PromptBuilder()
    prompt = builder.build_article_prompt(
        topic="GraphRAG",
        audience="Engineering Students",
        country="India",
        tone="Professional",
        length=1500,
        context=sample_context,
    )

    assert "## System Role" in prompt
    assert "## Task" in prompt
    assert "## Audience" in prompt
    assert "## Country" in prompt
    assert "## Tone" in prompt
    assert "## Target Length" in prompt
    assert "## Retrieved Vector Context" in prompt
    assert "## Knowledge Graph Context" in prompt
    assert "## Writing Instructions" in prompt
    assert "## Citation Instructions" in prompt
    assert "## Output Schema" in prompt
    assert "GraphRAG combines vector retrieval" in prompt
    assert "Knowledge Graph" in prompt
    assert "Indian English" in prompt


def test_prompt_builder_rejects_unsupported_country(sample_context: HybridContext):
    builder = PromptBuilder()
    with pytest.raises(ValueError, match="Unsupported country"):
        builder.build_article_prompt(
            topic="GraphRAG",
            audience="Students",
            country="Canada",
            tone="Professional",
            length=1000,
            context=sample_context,
        )


@pytest.mark.parametrize("country", sorted(SUPPORTED_COUNTRIES))
def test_supported_countries_have_localization(country: str, sample_context: HybridContext):
    builder = PromptBuilder()
    prompt = builder.build_article_prompt(
        topic="AI",
        audience="Students",
        country=country,
        tone="Professional",
        length=800,
        context=sample_context,
    )
    assert f"## Country\n{country}" in prompt
