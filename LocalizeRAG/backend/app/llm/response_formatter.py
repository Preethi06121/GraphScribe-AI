import json
import logging
import re

from pydantic import ValidationError

from app.schemas.content import (
    ArticleCitation,
    ArticleMetadata,
    ArticleSection,
    GeneratedArticleResponse,
    LLMArticlePayload,
)
from app.schemas.retrieval import HybridContext

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Formats LLM output and retrieval context into structured article JSON."""

    def format_article(
        self,
        raw_response: str,
        topic: str,
        audience: str,
        country: str,
        tone: str,
        target_length: int,
        context: HybridContext,
        explainability,
    ) -> GeneratedArticleResponse:
        payload = self._parse_llm_payload(raw_response)
        citations = self._build_citations(context)
        word_count = self._count_words(payload)

        return GeneratedArticleResponse(
            title=payload.title,
            summary=payload.summary,
            sections=payload.sections,
            conclusion=payload.conclusion,
            citations=citations,
            metadata=ArticleMetadata(
                retrieval_strategy="HybridRAG",
                country=country,
                audience=audience,
                tone=tone,
                topic=topic,
                word_count=word_count,
                target_length=target_length,
            ),
            explainability=explainability,
        )

    def _parse_llm_payload(self, raw_response: str) -> LLMArticlePayload:
        cleaned = self._extract_json(raw_response)
        try:
            data = json.loads(cleaned)
            return LLMArticlePayload.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Failed to parse LLM JSON response, using fallback formatter")
            return self._fallback_payload(raw_response)

    @staticmethod
    def _extract_json(raw_response: str) -> str:
        text = raw_response.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    @staticmethod
    def _fallback_payload(raw_response: str) -> LLMArticlePayload:
        paragraphs = [p.strip() for p in raw_response.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [raw_response.strip() or "Content could not be generated."]

        title = paragraphs[0][:120]
        summary = paragraphs[0]
        body = paragraphs[1:] or [paragraphs[0]]
        sections = [
            ArticleSection(heading=f"Section {index + 1}", content=content)
            for index, content in enumerate(body)
        ]
        conclusion = body[-1] if body else summary
        return LLMArticlePayload(
            title=title,
            summary=summary,
            sections=sections,
            conclusion=conclusion,
        )

    @staticmethod
    def _build_citations(context: HybridContext) -> list[ArticleCitation]:
        citations: list[ArticleCitation] = []
        seen: set[tuple[str, str, int | None, str]] = set()
        for citation in context.citations:
            key = (
                citation.document_id,
                citation.document_name,
                citation.page,
                citation.source,
            )
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                ArticleCitation(
                    document_id=citation.document_id,
                    document_name=citation.document_name,
                    page=citation.page,
                    source=citation.source,
                    reference=citation.reference,
                )
            )
        return citations

    @staticmethod
    def _count_words(payload: LLMArticlePayload) -> int:
        parts = [payload.title, payload.summary, payload.conclusion]
        parts.extend(section.content for section in payload.sections)
        return len(" ".join(parts).split())
