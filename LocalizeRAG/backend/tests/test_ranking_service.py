from app.retrieval.ranking import RankingService
from app.schemas.retrieval import GraphHit, VectorHit


def test_weighted_ranking_formula():
    ranking = RankingService(vector_weight=0.7, graph_weight=0.3)
    vector_hits = [
        VectorHit(
            chunk="chunk about OpenAI",
            score=1.0,
            page=1,
            document_id="doc-1",
            document_name="ai.pdf",
            chunk_id="c1",
        )
    ]
    graph_hits = [
        GraphHit(
            entity_name="OpenAI",
            entity_type="ORG",
            document_id="doc-1",
            document_name="ai.pdf",
            page_number=1,
            score=1.0,
        )
    ]

    ranked = ranking.rank(vector_hits, graph_hits)
    document_item = next(item for item in ranked if item.item_type == "document")

    assert abs(document_item.final_score - 1.0) < 1e-9
    assert ranking.vector_weight == 0.7
    assert ranking.graph_weight == 0.3


def test_configurable_weights():
    ranking = RankingService(vector_weight=0.4, graph_weight=0.6)
    assert abs(ranking.vector_weight - 0.4) < 1e-9
    assert abs(ranking.graph_weight - 0.6) < 1e-9


def test_no_duplicate_ranked_items():
    ranking = RankingService()
    vector_hits = [
        VectorHit(
            chunk="same",
            score=0.9,
            page=1,
            document_id="doc-1",
            chunk_id="c1",
        ),
        VectorHit(
            chunk="same",
            score=0.8,
            page=1,
            document_id="doc-1",
            chunk_id="c1",
        ),
    ]

    ranked = ranking.rank(vector_hits, [])
    document_items = [item for item in ranked if item.item_type == "document"]
    assert len(document_items) == 1
