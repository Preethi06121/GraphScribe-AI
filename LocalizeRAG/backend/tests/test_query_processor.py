import pytest

from app.retrieval.query_processor import QueryProcessor


def test_normalize_whitespace_and_case():
    processor = QueryProcessor()
    result = processor.process("  What is   GraphRAG?  ")

    assert result.normalized == "what is graphrag"
    assert result.semantic_query == "what is graphrag"
    assert "graphrag" in result.keywords
    assert "what" not in result.keywords
    assert "is" not in result.keywords


def test_empty_query_raises():
    processor = QueryProcessor()
    with pytest.raises(ValueError, match="empty"):
        processor.process("   ")


def test_keyword_extraction_deduplicates():
    processor = QueryProcessor()
    result = processor.process("RAG and rag with Embeddings embeddings")

    assert result.keywords.count("rag") == 1
    assert "embeddings" in result.keywords
