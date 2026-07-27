import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_hybrid_retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.retrieval import RetrievalSearchRequest, RetrievalSearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalSearchResponse)
async def search(
    request: RetrievalSearchRequest,
    hybrid_retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> RetrievalSearchResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    try:
        context = await hybrid_retriever.retrieve(query)
        return RetrievalSearchResponse(
            documents=context.documents,
            graph=context.graph,
            citations=context.citations,
            metadata=context.metadata,
        )
    except ValueError as exc:
        logger.warning("Invalid retrieval query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Hybrid retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while performing retrieval.",
        ) from exc
