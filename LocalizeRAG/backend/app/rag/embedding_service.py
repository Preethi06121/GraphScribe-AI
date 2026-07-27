import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates text embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded: %s", self._model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Cannot generate embeddings for an empty text list")

        logger.info("Generating embeddings for %d text(s)", len(texts))
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        logger.info("Generated %d embedding(s)", len(embeddings))
        return embeddings.tolist()
