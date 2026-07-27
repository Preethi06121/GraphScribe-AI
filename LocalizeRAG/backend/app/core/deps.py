import logging
from functools import lru_cache

from app.core.config import get_settings
from app.graph.entity_extractor import EntityExtractor
from app.graph.graph_builder import GraphBuilder
from app.graph.neo4j_service import Neo4jService
from app.graph.relationship_extractor import RelationshipExtractor
from app.llm.explainability import ExplainabilityGenerator
from app.llm.generation_engine import GenerationEngine
from app.llm.prompt_builder import PromptBuilder
from app.llm.provider_factory import create_provider
from app.llm.response_formatter import ResponseFormatter
from app.rag.document_loader import PDFDocumentLoader
from app.rag.embedding_service import EmbeddingService
from app.rag.ingestion_pipeline import IngestionPipeline
from app.rag.text_splitter import RecursiveTextSplitter
from app.rag.vector_store import VectorStore
from app.retrieval.context_fusion import ContextFusion
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_processor import QueryProcessor
from app.retrieval.ranking import RankingService
from app.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


@lru_cache
def get_document_loader() -> PDFDocumentLoader:
    return PDFDocumentLoader()


@lru_cache
def get_text_splitter() -> RecursiveTextSplitter:
    settings = get_settings()
    return RecursiveTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(model_name=settings.embedding_model)


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )


@lru_cache
def get_neo4j_service() -> Neo4jService:
    settings = get_settings()
    return Neo4jService(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )


@lru_cache
def get_entity_extractor() -> EntityExtractor:
    settings = get_settings()
    return EntityExtractor(model_name=settings.spacy_model)


@lru_cache
def get_relationship_extractor() -> RelationshipExtractor:
    return RelationshipExtractor()


@lru_cache
def get_query_processor() -> QueryProcessor:
    return QueryProcessor()


@lru_cache
def get_context_fusion() -> ContextFusion:
    return ContextFusion()


def get_vector_retriever() -> VectorRetriever:
    settings = get_settings()
    return VectorRetriever(
        vector_store=get_vector_store(),
        embedding_service=get_embedding_service(),
        top_k=settings.retrieval_top_k,
    )


def get_graph_retriever() -> GraphRetriever:
    settings = get_settings()
    return GraphRetriever(
        neo4j_service=get_neo4j_service(),
        entity_extractor=get_entity_extractor(),
        top_k=settings.retrieval_top_k,
    )


def get_ranking_service() -> RankingService:
    settings = get_settings()
    return RankingService(
        vector_weight=settings.retrieval_vector_weight,
        graph_weight=settings.retrieval_graph_weight,
    )


def get_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(
        query_processor=get_query_processor(),
        vector_retriever=get_vector_retriever(),
        graph_retriever=get_graph_retriever(),
        ranking_service=get_ranking_service(),
        context_fusion=get_context_fusion(),
        cache_size=settings.retrieval_cache_size,
        top_k=settings.retrieval_top_k,
    )


def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        document_loader=get_document_loader(),
        text_splitter=get_text_splitter(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


def get_graph_builder() -> GraphBuilder:
    return GraphBuilder(
        document_loader=get_document_loader(),
        entity_extractor=get_entity_extractor(),
        relationship_extractor=get_relationship_extractor(),
        neo4j_service=get_neo4j_service(),
    )


@lru_cache
def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder()


@lru_cache
def get_response_formatter() -> ResponseFormatter:
    return ResponseFormatter()


@lru_cache
def get_explainability_generator() -> ExplainabilityGenerator:
    return ExplainabilityGenerator()


def get_generation_engine() -> GenerationEngine:
    settings = get_settings()
    return GenerationEngine(
        hybrid_retriever=get_hybrid_retriever(),
        prompt_builder=get_prompt_builder(),
        provider=create_provider(settings),
        response_formatter=get_response_formatter(),
        explainability_generator=get_explainability_generator(),
    )
