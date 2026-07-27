from app.retrieval.context_fusion import ContextFusion
from app.retrieval.ranking import RankingService
from app.schemas.retrieval import GraphHit, ProcessedQuery, VectorHit


def test_context_fusion_structure_and_dedup():
    fusion = ContextFusion()
    ranking = RankingService()
    processed = ProcessedQuery(
        original="What is RAG?",
        normalized="what is rag",
        keywords=["rag"],
        semantic_query="what is rag",
    )
    vector_hits = [
        VectorHit(
            chunk="RAG combines retrieval and generation",
            score=0.9,
            page=1,
            document_id="doc-1",
            document_name="rag.pdf",
            chunk_id="c1",
            source="rag.pdf",
        ),
        VectorHit(
            chunk="RAG combines retrieval and generation",
            score=0.85,
            page=1,
            document_id="doc-1",
            document_name="rag.pdf",
            chunk_id="c1",
            source="rag.pdf",
        ),
    ]
    graph_hits = [
        GraphHit(
            entity_name="RAG",
            entity_type="TECH_TERM",
            connected_entity="Embeddings",
            relationship_type="RELATED_TO",
            document_id="doc-1",
            document_name="rag.pdf",
            page_number=1,
            score=0.8,
        )
    ]
    ranked = ranking.rank(vector_hits, graph_hits)

    context = fusion.fuse(
        processed,
        vector_hits,
        graph_hits,
        ranked,
        ranking.vector_weight,
        ranking.graph_weight,
    )

    assert len(context.documents) == 1
    assert len(context.graph) == 1
    assert len(context.citations) >= 1
    assert context.metadata.query == "What is RAG?"
    assert context.documents[0].source == "vector"
    assert context.graph[0].source == "graph"
    assert "documents" in context.model_dump()
    assert "graph" in context.model_dump()
    assert "citations" in context.model_dump()
    assert "metadata" in context.model_dump()
