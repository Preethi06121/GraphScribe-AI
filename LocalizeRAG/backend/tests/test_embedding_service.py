from unittest.mock import MagicMock, patch

import numpy as np

from app.rag.embedding_service import EmbeddingService


@patch("app.rag.embedding_service.SentenceTransformer")
def test_embed_texts_returns_embeddings(mock_transformer_cls):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_transformer_cls.return_value = mock_model

    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    embeddings = service.embed_texts(["hello world", "localizerag"])

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    mock_model.encode.assert_called_once_with(
        ["hello world", "localizerag"],
        normalize_embeddings=True,
    )


@patch("app.rag.embedding_service.SentenceTransformer")
def test_embed_texts_lazy_loads_model(mock_transformer_cls):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    mock_transformer_cls.return_value = mock_model

    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    service.embed_texts(["test"])

    mock_transformer_cls.assert_called_once_with("BAAI/bge-small-en-v1.5")
