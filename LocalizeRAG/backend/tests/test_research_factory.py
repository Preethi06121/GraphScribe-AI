from unittest.mock import MagicMock

import pytest

from app.research.factory import StrategyFactory
from app.research.strategies.graph_strategy import GraphStrategy
from app.research.strategies.hybrid_strategy import HybridStrategy
from app.research.strategies.vector_strategy import VectorStrategy


def test_factory_creates_vector_strategy():
    mock_vector = MagicMock()
    mock_graph = MagicMock()
    mock_hybrid = MagicMock()

    factory = StrategyFactory(
        vector_retriever=mock_vector,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
    )

    strategy = factory.get_strategy("vector")
    assert isinstance(strategy, VectorStrategy)
    assert strategy.name == "vector"

    # Case insensitivity test
    strategy_upper = factory.get_strategy("VECTOR")
    assert isinstance(strategy_upper, VectorStrategy)


def test_factory_creates_graph_strategy():
    mock_vector = MagicMock()
    mock_graph = MagicMock()
    mock_hybrid = MagicMock()

    factory = StrategyFactory(
        vector_retriever=mock_vector,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
    )

    strategy = factory.get_strategy("graph")
    assert isinstance(strategy, GraphStrategy)
    assert strategy.name == "graph"

    strategy_upper = factory.get_strategy("GRAPH")
    assert isinstance(strategy_upper, GraphStrategy)


def test_factory_creates_hybrid_strategy():
    mock_vector = MagicMock()
    mock_graph = MagicMock()
    mock_hybrid = MagicMock()

    factory = StrategyFactory(
        vector_retriever=mock_vector,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
    )

    strategy = factory.get_strategy("hybrid")
    assert isinstance(strategy, HybridStrategy)
    assert strategy.name == "hybrid"

    strategy_upper = factory.get_strategy("HYBRID")
    assert isinstance(strategy_upper, HybridStrategy)


def test_factory_unknown_strategy_raises():
    mock_vector = MagicMock()
    mock_graph = MagicMock()
    mock_hybrid = MagicMock()

    factory = StrategyFactory(
        vector_retriever=mock_vector,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
    )

    with pytest.raises(ValueError, match="Unknown retrieval strategy"):
        factory.get_strategy("invalid_strategy")


def test_factory_supported_strategies_list():
    strategies = StrategyFactory.get_supported_strategies()
    assert "vector" in strategies
    assert "graph" in strategies
    assert "hybrid" in strategies
