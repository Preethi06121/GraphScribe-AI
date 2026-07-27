from app.graph.entity_extractor import EntityExtractor
from app.graph.relationship_extractor import RelationshipExtractor
from app.schemas.graph import GraphEntity


def test_extract_explicit_relationship():
    extractor = EntityExtractor(model_name="en_core_web_sm")
    relationship_extractor = RelationshipExtractor()

    text = "OpenAI developed GPT-4."
    document_id = "doc-rel-1"
    entities = [
        GraphEntity(
            entity_name="OpenAI",
            entity_type="ORG",
            document_id=document_id,
            document_name="research.pdf",
            page_number=1,
        ),
        GraphEntity(
            entity_name="GPT-4",
            entity_type="PRODUCT",
            document_id=document_id,
            document_name="research.pdf",
            page_number=1,
        ),
    ]

    nlp_doc = extractor.get_nlp_doc(text)
    relationships = relationship_extractor.extract(
        text,
        entities,
        document_id,
        "research.pdf",
        1,
        nlp_doc,
    )

    assert len(relationships) >= 1
    rel_types = {rel.relationship_type for rel in relationships}
    assert "DEVELOPED" in rel_types or "RELATED_TO" in rel_types
    assert relationships[0].document_id == document_id
    assert relationships[0].source_document == "research.pdf"


def test_co_occurrence_related_to():
    relationship_extractor = RelationshipExtractor()
    extractor = EntityExtractor(model_name="en_core_web_sm")

    text = "Apple and Google attended the summit."
    document_id = "doc-rel-2"
    entities = [
        GraphEntity(
            entity_name="Apple",
            entity_type="ORG",
            document_id=document_id,
            document_name="report.pdf",
            page_number=1,
        ),
        GraphEntity(
            entity_name="Google",
            entity_type="ORG",
            document_id=document_id,
            document_name="report.pdf",
            page_number=1,
        ),
    ]

    nlp_doc = extractor.get_nlp_doc(text)
    relationships = relationship_extractor.extract(
        text,
        entities,
        document_id,
        "report.pdf",
        1,
        nlp_doc,
    )

    assert len(relationships) >= 1
    assert any(rel.relationship_type == "RELATED_TO" for rel in relationships)
