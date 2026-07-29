import logging

from app.llm.explainability import ExplainabilityGenerator
from app.llm.prompt_builder import PromptBuilder
from app.llm.provider import LLMProvider, LLMProviderError
from app.llm.response_formatter import ResponseFormatter
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.content import GeneratedArticleResponse

logger = logging.getLogger(__name__)


from app.research.factory import StrategyFactory


class GenerationEngine:
    """Orchestrates strategy-aware retrieval-grounded long-form article generation."""

    def __init__(
        self,
        strategy_factory: StrategyFactory | HybridRetriever | None = None,
        prompt_builder: PromptBuilder | None = None,
        provider: LLMProvider | None = None,
        response_formatter: ResponseFormatter | None = None,
        explainability_generator: ExplainabilityGenerator | None = None,
        hybrid_retriever: HybridRetriever | None = None,
    ) -> None:
        if isinstance(strategy_factory, StrategyFactory):
            self._strategy_factory = strategy_factory
            self._hybrid_retriever = hybrid_retriever
        elif strategy_factory is not None:
            self._strategy_factory = None
            self._hybrid_retriever = strategy_factory
        else:
            self._strategy_factory = None
            self._hybrid_retriever = hybrid_retriever

        self._prompt_builder = prompt_builder or PromptBuilder()
        self._provider = provider
        self._response_formatter = response_formatter or ResponseFormatter()
        self._explainability_generator = explainability_generator or ExplainabilityGenerator()

    async def generate_article(
        self,
        topic: str,
        audience: str,
        country: str,
        tone: str,
        length: int,
        retrieval_strategy: str = "HYBRID",
    ) -> GeneratedArticleResponse:
        logger.info(
            "Starting article generation: topic='%s' audience='%s' country='%s' strategy='%s'",
            topic,
            audience,
            country,
            retrieval_strategy,
        )

        retrieval_query = self._build_retrieval_query(topic, audience, country)
        context = await self._retrieve_context(retrieval_query, retrieval_strategy)
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

    async def _retrieve_context(self, query: str, strategy_name: str):
        if self._strategy_factory:
            strategy = self._strategy_factory.get_strategy(strategy_name)
            if hasattr(strategy, "retrieve_context"):
                return await strategy.retrieve_context(query)
            if self._hybrid_retriever:
                return await self._hybrid_retriever.retrieve(query)
            raise ValueError(f"Unable to retrieve context for strategy '{strategy_name}'")
        elif self._hybrid_retriever:
            return await self._hybrid_retriever.retrieve(query)
        else:
            raise ValueError("No strategy factory or retriever configured.")

    @staticmethod
    def _build_retrieval_query(topic: str, audience: str, country: str) -> str:
        return f"{topic} for {audience} in {country}"
