import json

from app.llm.explainability import ExplainabilityGenerator
from app.llm.response_formatter import ResponseFormatter
from app.schemas.retrieval import (
    ContextDocument,
    ContextGraphItem,
    Citation,
    HybridContext,
    HybridContextMetadata,
)


def _sample_context() -> HybridContext:
    return HybridContext(
        documents=[
            ContextDocument(
                chunk="RAG improves answer grounding.",
                score=0.9,
                page=2,
                document_id="doc-1",
                document_name="rag.pdf",
                chunk_id="c1",
            )
        ],
        graph=[
            ContextGraphItem(
                entity_name="RAG",
                entity_type="TECH_TERM",
                document_id="doc-1",
                document_name="rag.pdf",
                page_number=2,
                score=0.75,
            )
        ],
        citations=[
            Citation(
                document_id="doc-1",
                document_name="rag.pdf",
                page=2,
                source="vector",
                reference="c1",
            )
        ],
        metadata=HybridContextMetadata(
            query="rag",
            normalized_query="rag",
            keywords=["rag"],
            vector_hits=1,
            graph_hits=1,
            vector_weight=0.7,
            graph_weight=0.3,
        ),
    )


def test_response_formatter_parses_json():
    formatter = ResponseFormatter()
    explainability = ExplainabilityGenerator().generate(_sample_context())
    payload = {
        "title": "Understanding RAG",
        "summary": "A concise overview of retrieval-augmented generation.",
        "sections": [
            {"heading": "Introduction", "content": "RAG combines retrieval and generation."}
        ],
        "conclusion": "RAG is a powerful architecture.",
    }

    article = formatter.format_article(
        raw_response=json.dumps(payload),
        topic="RAG",
        audience="Students",
        country="India",
        tone="Professional",
        target_length=1000,
        context=_sample_context(),
        explainability=explainability,
    )

    assert article.title == "Understanding RAG"
    assert len(article.sections) == 1
    assert article.metadata.retrieval_strategy == "HybridRAG"
    assert article.metadata.country == "India"
    assert article.citations[0].document_id == "doc-1"
    assert article.explainability.chunk_count == 1


def test_response_formatter_handles_markdown_fenced_json():
    formatter = ResponseFormatter()
    explainability = ExplainabilityGenerator().generate(_sample_context())
    raw = """```json
{
  "title": "GraphRAG Overview",
  "summary": "Summary text",
  "sections": [{"heading": "Basics", "content": "GraphRAG content"}],
  "conclusion": "Final thoughts"
}
```"""
    article = formatter.format_article(
        raw_response=raw,
        topic="GraphRAG",
        audience="Engineers",
        country="United States",
        tone="Professional",
        target_length=1200,
        context=_sample_context(),
        explainability=explainability,
    )

    assert article.title == "GraphRAG Overview"
    assert article.sections[0].heading == "Basics"
