import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_comparison_service
from app.research.comparison_service import ComparisonService
from app.research.schemas import (
    ResearchComparisonRequest,
    ResearchComparisonResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/compare", response_model=ResearchComparisonResponse)
async def compare_strategies(
    request: ResearchComparisonRequest,
    comparison_service: ComparisonService = Depends(get_comparison_service),
) -> ResearchComparisonResponse:
    """Compare performance metrics across vector, graph, and hybrid retrieval strategies."""
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    try:
        return await comparison_service.compare(request)
    except ValueError as exc:
        logger.warning("Invalid strategy comparison request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Strategy comparison failed unexpectedly")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while comparing retrieval strategies.",
        ) from exc
