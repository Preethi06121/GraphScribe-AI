import logging

from app.schemas.content import (
    DocumentUsed,
    Explainability,
    GraphEntityExplainability,
    VectorChunkExplainability,
)
from app.schemas.retrieval import HybridContext

logger = logging.getLogger(__name__)


class ExplainabilityGenerator:
    """Generates retrieval explainability metadata without using an LLM."""

    def generate(self, context: HybridContext) -> Explainability:
        vector_chunks = [
            VectorChunkExplainability(
                chunk_id=doc.chunk_id,
                document_id=doc.document_id,
                document_name=doc.document_name,
                page=doc.page,
                score=doc.score,
                excerpt=doc.chunk[:240],
            )
            for doc in context.documents
        ]

        graph_entities = [
            GraphEntityExplainability(
                entity_name=item.entity_name,
                entity_type=item.entity_type,
                document_id=item.document_id,
                document_name=item.document_name,
                page_number=item.page_number,
                score=item.score,
                relationship_type=item.relationship_type,
                connected_entity=item.connected_entity,
            )
            for item in context.graph
        ]

        documents_used = self._aggregate_documents(context)
        vector_score_summary = self._average_score([doc.score for doc in context.documents])
        graph_score_summary = self._average_score([item.score for item in context.graph])

        reasoning = self._build_reasoning(
            context,
            vector_score_summary,
            graph_score_summary,
            len(vector_chunks),
            len(graph_entities),
        )

        explainability = Explainability(
            vector_chunks=vector_chunks,
            graph_entities=graph_entities,
            reasoning=reasoning,
            documents_used=documents_used,
            retrieval_strategy="HybridRAG",
            vector_score_summary=vector_score_summary,
            graph_score_summary=graph_score_summary,
            chunk_count=len(vector_chunks),
            entity_count=len(graph_entities),
        )
        logger.info(
            "Generated explainability metadata: %d chunks, %d entities, %d documents",
            explainability.chunk_count,
            explainability.entity_count,
            len(documents_used),
        )
        return explainability

    @staticmethod
    def _aggregate_documents(context: HybridContext) -> list[DocumentUsed]:
        documents: dict[str, DocumentUsed] = {}

        for doc in context.documents:
            if doc.document_id not in documents:
                documents[doc.document_id] = DocumentUsed(
                    document_id=doc.document_id,
                    document_name=doc.document_name,
                    source_types=[],
                )
            if "vector" not in documents[doc.document_id].source_types:
                documents[doc.document_id].source_types.append("vector")

        for item in context.graph:
            if item.document_id not in documents:
                documents[item.document_id] = DocumentUsed(
                    document_id=item.document_id,
                    document_name=item.document_name,
                    source_types=[],
                )
            if "graph" not in documents[item.document_id].source_types:
                documents[item.document_id].source_types.append("graph")

        return list(documents.values())

    @staticmethod
    def _average_score(scores: list[float]) -> float:
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 4)

    @staticmethod
    def _build_reasoning(
        context: HybridContext,
        vector_score_summary: float,
        graph_score_summary: float,
        chunk_count: int,
        entity_count: int,
    ) -> str:
        if chunk_count == 0 and entity_count == 0:
            return (
                "No retrieved vector chunks or graph entities were available for this query. "
                "Generation relied on prompt instructions and topic knowledge only."
            )

        parts = [
            "HybridRAG selected context using weighted vector and graph retrieval.",
            f"Vector retrieval contributed {chunk_count} chunk(s) "
            f"with average score {vector_score_summary:.3f}.",
            f"Graph retrieval contributed {entity_count} entit(y/ies) "
            f"with average score {graph_score_summary:.3f}.",
        ]

        if context.metadata.keywords:
            parts.append(
                "Keywords used during retrieval: "
                + ", ".join(context.metadata.keywords)
                + "."
            )

        if chunk_count and entity_count:
            parts.append(
                "Both document chunks and knowledge graph entities informed the article structure."
            )
        elif chunk_count:
            parts.append("Document chunks were the primary grounding source.")
        else:
            parts.append("Knowledge graph entities were the primary grounding source.")

        return " ".join(parts)
