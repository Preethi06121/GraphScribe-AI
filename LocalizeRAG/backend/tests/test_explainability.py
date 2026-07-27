from app.llm.explainability import ExplainabilityGenerator
from app.schemas.retrieval import (
    ContextDocument,
    ContextGraphItem,
    Citation,
    HybridContext,
    HybridContextMetadata,
)


def test_explainability_generator_metadata():
    context = HybridContext(
        documents=[
            ContextDocument(
                chunk="Hybrid retrieval combines vector and graph signals.",
                score=0.88,
                page=1,
                document_id="doc-1",
                document_name="guide.pdf",
                chunk_id="c1",
            ),
            ContextDocument(
                chunk="Another supporting chunk.",
                score=0.76,
                page=3,
                document_id="doc-2",
                document_name="notes.pdf",
                chunk_id="c2",
            ),
        ],
        graph=[
            ContextGraphItem(
                entity_name="GraphRAG",
                entity_type="TECH_TERM",
                connected_entity="Embeddings",
                relationship_type="RELATED_TO",
                document_id="doc-1",
                document_name="guide.pdf",
                page_number=1,
                score=0.7,
            )
        ],
        citations=[],
        metadata=HybridContextMetadata(
            query="graphrag embeddings",
            normalized_query="graphrag embeddings",
            keywords=["graphrag", "embeddings"],
            vector_hits=2,
            graph_hits=1,
            vector_weight=0.7,
            graph_weight=0.3,
        ),
    )

    explainability = ExplainabilityGenerator().generate(context)

    assert explainability.retrieval_strategy == "HybridRAG"
    assert explainability.chunk_count == 2
    assert explainability.entity_count == 1
    assert explainability.vector_score_summary == 0.82
    assert explainability.graph_score_summary == 0.7
    assert len(explainability.documents_used) == 2
    assert "HybridRAG" in explainability.reasoning
    assert explainability.vector_chunks[0].excerpt.startswith("Hybrid retrieval")


def test_explainability_empty_context_reasoning():
    context = HybridContext(
        documents=[],
        graph=[],
        citations=[],
        metadata=HybridContextMetadata(
            query="empty",
            normalized_query="empty",
            keywords=[],
            vector_hits=0,
            graph_hits=0,
            vector_weight=0.7,
            graph_weight=0.3,
        ),
    )

    explainability = ExplainabilityGenerator().generate(context)

    assert explainability.chunk_count == 0
    assert explainability.entity_count == 0
    assert "No retrieved vector chunks" in explainability.reasoning
