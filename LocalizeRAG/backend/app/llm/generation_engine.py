import logging

from app.llm.explainability import ExplainabilityGenerator
from app.llm.prompt_builder import PromptBuilder
from app.llm.provider import LLMProvider, LLMProviderError
from app.llm.response_formatter import ResponseFormatter
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.content import GeneratedArticleResponse

logger = logging.getLogger(__name__)


class GenerationEngine:
    """Orchestrates retrieval-grounded long-form article generation."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        prompt_builder: PromptBuilder,
        provider: LLMProvider,
        response_formatter: ResponseFormatter,
        explainability_generator: ExplainabilityGenerator,
    ) -> None:
        self._hybrid_retriever = hybrid_retriever
        self._prompt_builder = prompt_builder
        self._provider = provider
        self._response_formatter = response_formatter
        self._explainability_generator = explainability_generator

    async def generate_article(
        self,
        topic: str,
        audience: str,
        country: str,
        tone: str,
        length: int,
    ) -> GeneratedArticleResponse:
        logger.info(
            "Starting article generation: topic='%s' audience='%s' country='%s'",
            topic,
            audience,
            country,
        )

        retrieval_query = self._build_retrieval_query(topic, audience, country)
        context = await self._hybrid_retriever.retrieve(retrieval_query)
        explainability = self._explainability_generator.generate(context)

        prompt = self._prompt_builder.build_article_prompt(
            topic=topic,
            audience=audience,
            country=country,
            tone=tone,
            length=length,
            context=context,
        )

        try:
            raw_response = await self._provider.generate(prompt)
        except LLMProviderError as exc:
            logger.error("LLM provider failed during article generation: %s", exc)
            raise

        article = self._response_formatter.format_article(
            raw_response=raw_response,
            topic=topic,
            audience=audience,
            country=country,
            tone=tone,
            target_length=length,
            context=context,
            explainability=explainability,
        )

        logger.info(
            "Article generation complete: title='%s' word_count=%d",
            article.title,
            article.metadata.word_count,
        )
        return article

    @staticmethod
    def _build_retrieval_query(topic: str, audience: str, country: str) -> str:
        return f"{topic} for {audience} in {country}"
