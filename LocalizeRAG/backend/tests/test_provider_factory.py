import pytest

from app.core.config import Settings
from app.llm.provider import LLMProviderError, OllamaProvider, OpenRouterProvider
from app.llm.provider_factory import create_provider


def test_provider_factory_creates_ollama_provider():
    settings = Settings(llm_provider="ollama", ollama_model="llama3.2")
    provider = create_provider(settings)
    assert isinstance(provider, OllamaProvider)


def test_provider_factory_creates_openrouter_provider():
    settings = Settings(
        llm_provider="openrouter",
        openrouter_api_key="test-key",
        openrouter_model="openai/gpt-4o-mini",
    )
    provider = create_provider(settings)
    assert isinstance(provider, OpenRouterProvider)


def test_provider_factory_rejects_unknown_provider():
    settings = Settings(llm_provider="unknown")
    with pytest.raises(LLMProviderError, match="Unsupported LLM provider"):
        create_provider(settings)


def test_openrouter_requires_api_key():
    settings = Settings(llm_provider="openrouter", openrouter_api_key="")
    with pytest.raises(LLMProviderError, match="OPENROUTER_API_KEY"):
        create_provider(settings)
