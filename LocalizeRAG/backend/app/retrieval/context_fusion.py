import logging

from app.schemas.retrieval import (
    Citation,
    ContextDocument,
    ContextGraphItem,
    GraphHit,
    HybridContext,
    HybridContextMetadata,
    ProcessedQuery,
    RankedItem,
    VectorHit,
)

logger = logging.getLogger(__name__)


class ContextFusion:
    """Merges vector chunks and graph entities into a structured context."""

    def fuse(
        self,
        processed_query: ProcessedQuery,
        vector_hits: list[VectorHit],
        graph_hits: list[GraphHit],
        ranked_items: list[RankedItem],
        vector_weight: float,
        graph_weight: float,
        top_documents: int = 5,
        top_graph: int = 5,
    ) -> HybridContext:
        documents = self._build_documents(vector_hits, ranked_items, top_documents)
        graph_items = self._build_graph(graph_hits, top_graph)
        citations = self._build_citations(documents, graph_items)

        context = HybridContext(
            documents=documents,
            graph=graph_items,
            citations=citations,
            metadata=HybridContextMetadata(
                query=processed_query.original,
                normalized_query=processed_query.normalized,
                keywords=processed_query.keywords,
                vector_hits=len(vector_hits),
                graph_hits=len(graph_hits),
                vector_weight=vector_weight,
                graph_weight=graph_weight,
            ),
        )
        logger.info(
            "Fused context: %d document(s), %d graph item(s), %d citation(s)",
            len(documents),
            len(graph_items),
            len(citations),
        )
        return context

    def _build_documents(
        self,
        vector_hits: list[VectorHit],
        ranked_items: list[RankedItem],
        top_documents: int,
    ) -> list[ContextDocument]:
        score_by_chunk: dict[str, float] = {}
        for item in ranked_items:
            if item.item_type != "document":
                continue
            score_by_chunk[item.item_id] = item.final_score

        documents: list[ContextDocument] = []
        seen_chunks: set[str] = set()

        ordered_hits = sorted(
            vector_hits,
            key=lambda hit: score_by_chunk.get(
                hit.chunk_id or f"{hit.document_id}:{hit.page}:{hash(hit.chunk)}",
                hit.score,
            ),
            reverse=True,
        )

        for hit in ordered_hits:
            dedupe_key = hit.chunk_id or f"{hit.document_id}:{hit.page}:{hit.chunk[:80]}"
            if dedupe_key in seen_chunks:
                continue
            seen_chunks.add(dedupe_key)

            item_id = hit.chunk_id or f"{hit.document_id}:{hit.page}:{hash(hit.chunk)}"
            documents.append(
                ContextDocument(
                    chunk=hit.chunk,
                    score=score_by_chunk.get(item_id, hit.score),
                    page=hit.page,
                    document_id=hit.document_id,
                    document_name=hit.document_name,
                    chunk_id=hit.chunk_id,
                    source="vector",
                )
            )
            if len(documents) >= top_documents:
                break

        return documents

    def _build_graph(self, graph_hits: list[GraphHit], top_graph: int) -> list[ContextGraphItem]:
        items: list[ContextGraphItem] = []
        seen: set[tuple[str, str, str | None, str | None]] = set()

        for hit in graph_hits:
            key = (
                hit.entity_name,
                hit.document_id,
                hit.connected_entity,
                hit.relationship_type,
            )
            if key in seen:
                continue
            seen.add(key)
            items.append(
                ContextGraphItem(
                    entity_name=hit.entity_name,
                    entity_type=hit.entity_type,
                    connected_entity=hit.connected_entity,
                    relationship_type=hit.relationship_type,
                    document_id=hit.document_id,
                    document_name=hit.document_name,
                    page_number=hit.page_number,
                    score=hit.score,
                    source="graph",
                )
            )
            if len(items) >= top_graph:
                break

        return items

    def _build_citations(
        self,
        documents: list[ContextDocument],
        graph_items: list[ContextGraphItem],
    ) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, str, int | None, str]] = set()

        for doc in documents:
            key = (doc.document_id, doc.document_name, doc.page, "vector")
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    document_id=doc.document_id,
                    document_name=doc.document_name,
                    page=doc.page,
                    source="vector",
                    reference=doc.chunk_id or f"page-{doc.page}",
                )
            )

        for item in graph_items:
            key = (item.document_id, item.document_name, item.page_number or None, "graph")
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    document_id=item.document_id,
                    document_name=item.document_name,
                    page=item.page_number or None,
                    source="graph",
                    reference=item.entity_name,
                )
            )

        return citations
