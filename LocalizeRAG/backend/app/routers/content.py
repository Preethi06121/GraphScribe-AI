import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_generation_engine
from app.llm.generation_engine import GenerationEngine
from app.llm.provider import LLMProviderError
from app.schemas.content import ContentGenerateRequest, GeneratedArticleResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/content", tags=["content"])


@router.post("/generate", response_model=GeneratedArticleResponse)
async def generate_content(
    request: ContentGenerateRequest,
    generation_engine: GenerationEngine = Depends(get_generation_engine),
) -> GeneratedArticleResponse:
    strat = (request.retrieval_strategy or "HYBRID").strip().upper()
    if strat not in {"VECTOR", "GRAPH", "HYBRID"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid retrieval strategy '{request.retrieval_strategy}'. Supported values are: VECTOR, GRAPH, HYBRID.",
        )

    try:
        return await generation_engine.generate_article(
            topic=request.topic.strip(),
            audience=request.audience.strip(),
            country=request.country.strip(),
            tone=request.tone.strip(),
            length=request.length,
            retrieval_strategy=strat,
        )
    except ValueError as exc:
        logger.warning("Invalid content generation request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except LLMProviderError as exc:
        logger.error("LLM provider error during content generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Content generation service is currently unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during content generation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating content.",
        ) from exc
