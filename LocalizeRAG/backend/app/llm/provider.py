import abc
import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when an LLM provider request fails."""


class LLMProvider(abc.ABC):
    """Abstract interface for LLM content generation providers."""

    @abc.abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt and return the raw response string."""


class OpenRouterProvider(LLMProvider):
    """OpenRouter chat completions provider."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise LLMProviderError("OPENROUTER_API_KEY is not configured")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    async def generate(self, prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }
        return await self._post_with_retry(url, headers, payload)

    async def _post_with_retry(
        self,
        url: str,
        headers: dict,
        payload: dict,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    logger.info("OpenRouter generation succeeded on attempt %d", attempt)
                    return str(content)
            except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
                logger.warning(
                    "OpenRouter generation failed on attempt %d: %s",
                    attempt,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))
        raise LLMProviderError(
            f"OpenRouter generation failed after {self._max_retries} attempts"
        ) from last_error


class OllamaProvider(LLMProvider):
    """Ollama local chat provider."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    async def generate(self, prompt: str) -> str:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
        }
        return await self._post_with_retry(url, payload)

    async def _post_with_retry(self, url: str, payload: dict) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    content = data["message"]["content"]
                    logger.info("Ollama generation succeeded on attempt %d", attempt)
                    return str(content)
            except (httpx.HTTPError, KeyError, TypeError) as exc:
                last_error = exc
                logger.warning(
                    "Ollama generation failed on attempt %d: %s",
                    attempt,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))
        raise LLMProviderError(
            f"Ollama generation failed after {self._max_retries} attempts"
        ) from last_error
