import logging

from app.schemas.retrieval import GraphHit, RankedItem, VectorHit

logger = logging.getLogger(__name__)


class RankingService:
    """Weighted hybrid ranking of vector and graph retrieval results."""

    def __init__(self, vector_weight: float = 0.7, graph_weight: float = 0.3) -> None:
        total = vector_weight + graph_weight
        if total <= 0:
            raise ValueError("Ranking weights must sum to a positive value")
        self._vector_weight = vector_weight / total
        self._graph_weight = graph_weight / total

    @property
    def vector_weight(self) -> float:
        return self._vector_weight

    @property
    def graph_weight(self) -> float:
        return self._graph_weight

    def rank(
        self,
        vector_hits: list[VectorHit],
        graph_hits: list[GraphHit],
    ) -> list[RankedItem]:
        graph_by_document = self._aggregate_graph_scores(graph_hits)
        ranked: list[RankedItem] = []
        seen_ids: set[str] = set()

        for hit in vector_hits:
            item_id = hit.chunk_id or f"{hit.document_id}:{hit.page}:{hash(hit.chunk)}"
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            graph_score = graph_by_document.get(hit.document_id, 0.0)
            final_score = (
                self._vector_weight * hit.score + self._graph_weight * graph_score
            )
            ranked.append(
                RankedItem(
                    item_id=item_id,
                    item_type="document",
                    content=hit.chunk,
                    document_id=hit.document_id,
                    document_name=hit.document_name,
                    page=hit.page,
                    vector_score=hit.score,
                    graph_score=graph_score,
                    final_score=final_score,
                    source="vector",
                    metadata={"chunk_id": hit.chunk_id, "source": hit.source},
                )
            )

        for hit in graph_hits:
            item_id = (
                f"graph:{hit.document_id}:{hit.entity_name}:"
                f"{hit.connected_entity}:{hit.relationship_type}"
            )
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            vector_score = 0.0
            final_score = (
                self._vector_weight * vector_score + self._graph_weight * hit.score
            )
            content = hit.entity_name
            if hit.relationship_type and hit.connected_entity:
                content = f"{hit.entity_name} -[{hit.relationship_type}]-> {hit.connected_entity}"

            ranked.append(
                RankedItem(
                    item_id=item_id,
                    item_type="graph",
                    content=content,
                    document_id=hit.document_id,
                    document_name=hit.document_name,
                    page=hit.page_number,
                    vector_score=vector_score,
                    graph_score=hit.score,
                    final_score=final_score,
                    source="graph",
                    metadata={
                        "entity_name": hit.entity_name,
                        "entity_type": hit.entity_type,
                        "connected_entity": hit.connected_entity,
                        "relationship_type": hit.relationship_type,
                    },
                )
            )

        ranked.sort(key=lambda item: item.final_score, reverse=True)
        logger.info("Ranked %d item(s)", len(ranked))
        return ranked

    @staticmethod
    def _aggregate_graph_scores(graph_hits: list[GraphHit]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for hit in graph_hits:
            if not hit.document_id:
                continue
            current = scores.get(hit.document_id, 0.0)
            scores[hit.document_id] = max(current, hit.score)
        return scores
