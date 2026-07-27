import logging

from app.core.config import Settings
from app.llm.provider import LLMProvider, LLMProviderError, OllamaProvider, OpenRouterProvider

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = frozenset({"openrouter", "ollama"})


def create_provider(settings: Settings) -> LLMProvider:
    """Create an LLM provider based on application settings."""
    provider_name = settings.llm_provider.strip().lower()

    if provider_name not in SUPPORTED_PROVIDERS:
        raise LLMProviderError(
            f"Unsupported LLM provider '{settings.llm_provider}'. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )

    if provider_name == "openrouter":
        logger.info("Initializing OpenRouter provider with model %s", settings.openrouter_model)
        return OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    logger.info("Initializing Ollama provider with model %s", settings.ollama_model)
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
