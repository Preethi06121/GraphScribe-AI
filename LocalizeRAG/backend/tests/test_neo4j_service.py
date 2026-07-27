from unittest.mock import MagicMock, patch

from app.graph.neo4j_service import Neo4jService
from app.schemas.graph import GraphEntity, GraphRelationship


@patch("app.graph.neo4j_service.GraphDatabase")
def test_merge_entities(mock_graph_database):
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = {"count": 2}
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_graph_database.driver.return_value = mock_driver

    service = Neo4jService("bolt://localhost:7687", "neo4j", "password")
    entities = [
        GraphEntity(
            entity_name="OpenAI",
            entity_type="ORG",
            document_id="doc-1",
            document_name="test.pdf",
            page_number=1,
        ),
        GraphEntity(
            entity_name="GPT-4",
            entity_type="PRODUCT",
            document_id="doc-1",
            document_name="test.pdf",
            page_number=1,
        ),
    ]

    count = service.merge_entities(entities)

    assert count == 2
    mock_session.run.assert_called()


@patch("app.graph.neo4j_service.GraphDatabase")
def test_merge_relationships(mock_graph_database):
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = {"count": 1}
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_graph_database.driver.return_value = mock_driver

    service = Neo4jService("bolt://localhost:7687", "neo4j", "password")
    relationships = [
        GraphRelationship(
            source_entity="OpenAI",
            source_type="ORG",
            relationship_type="DEVELOPED",
            target_entity="GPT-4",
            target_type="PRODUCT",
            document_id="doc-1",
            source_document="test.pdf",
            page_number=1,
        )
    ]

    count = service.merge_relationships(relationships)

    assert count == 1
    mock_session.run.assert_called()


@patch("app.graph.neo4j_service.GraphDatabase")
def test_get_statistics(mock_graph_database):
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = {
        "node_count": 10,
        "rel_count": 5,
        "entity_types": ["ORG", "PERSON"],
        "document_count": 2,
    }
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_graph_database.driver.return_value = mock_driver

    service = Neo4jService("bolt://localhost:7687", "neo4j", "password")
    stats = service.get_statistics()

    assert stats.nodes == 10
    assert stats.relationships == 5
    assert stats.documents == 2
    assert stats.entity_types == ["ORG", "PERSON"]
