import logging

from fastapi import APIRouter, Depends, HTTPException, status
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.deps import get_neo4j_service
from app.graph.neo4j_service import Neo4jService
from app.schemas.graph import DocumentGraphResponse, GraphStatistics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/statistics", response_model=GraphStatistics)
async def get_graph_statistics(
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
) -> GraphStatistics:
    if not neo4j_service.is_available():
        logger.warning("Neo4j unavailable for graph statistics request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j database is currently unavailable.",
        )

    try:
        return neo4j_service.get_statistics()
    except (ServiceUnavailable, Neo4jError, OSError) as exc:
        logger.warning("Failed to fetch graph statistics: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve graph statistics.",
        ) from exc


@router.get("/document/{document_id}", response_model=DocumentGraphResponse)
async def get_document_graph(
    document_id: str,
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
) -> DocumentGraphResponse:
    if not neo4j_service.is_available():
        logger.warning("Neo4j unavailable for document graph request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j database is currently unavailable.",
        )

    try:
        graph = neo4j_service.get_document_graph(document_id)
    except (ServiceUnavailable, Neo4jError, OSError) as exc:
        logger.warning("Failed to fetch graph for document %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve document graph.",
        ) from exc

    if not graph.nodes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No graph data found for document_id: {document_id}",
        )

    return graph
