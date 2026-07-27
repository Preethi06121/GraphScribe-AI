import pytest

from app.graph.entity_extractor import EntityExtractor
from app.graph.graph_schema import SPACY_ENTITY_TYPES


@pytest.fixture(scope="module")
def extractor() -> EntityExtractor:
    return EntityExtractor(model_name="en_core_web_sm")


def test_extract_spacy_entities(extractor: EntityExtractor):
    text = "OpenAI is an ORG based in San Francisco. Elon Musk met with engineers."
    entities = extractor.extract_from_text(
        text,
        document_id="doc-1",
        document_name="test.pdf",
        page_number=1,
    )

    entity_types = {entity.entity_type for entity in entities}
    entity_names = {entity.entity_name for entity in entities}

    assert "ORG" in entity_types or "PERSON" in entity_types
    assert all(entity.document_id == "doc-1" for entity in entities)
    assert all(entity.page_number == 1 for entity in entities)


def test_extract_rule_based_tech_terms(extractor: EntityExtractor):
    text = (
        "This document discusses GraphRAG, RAG, LLM, Transformer, Embeddings, "
        "Vector Database, Knowledge Graph, Machine Learning, Deep Learning, "
        "Neural Network, Fine-tuning, and LoRA."
    )
    entities = extractor.extract_from_text(
        text,
        document_id="doc-2",
        document_name="ai.pdf",
        page_number=2,
    )

    tech_entities = [e for e in entities if e.entity_type == "TECH_TERM"]
    tech_names = {e.entity_name.lower() for e in tech_entities}

    assert "graphrag" in tech_names or "rag" in tech_names
    assert "llm" in tech_names
    assert len(tech_entities) >= 5


def test_entity_types_are_valid(extractor: EntityExtractor):
    text = "Microsoft announced a new PRODUCT at an EVENT in London."
    entities = extractor.extract_from_text(
        text,
        document_id="doc-3",
        document_name="news.pdf",
        page_number=1,
    )

    allowed_types = SPACY_ENTITY_TYPES | {"TECH_TERM"}
    assert all(entity.entity_type in allowed_types for entity in entities)
