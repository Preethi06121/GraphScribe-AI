from fastapi import APIRouter, HTTPException
from app.research.comparison_service import ComparisonService
from app.research.schemas import ResearchComparisonRequest, ResearchComparisonResponse

router = APIRouter()


@router.post("/research/compare", response_model=ResearchComparisonResponse, tags=["research"])
async def compare(request: ResearchComparisonRequest):
    service = ComparisonService()
    try:
        return await service.compare(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
