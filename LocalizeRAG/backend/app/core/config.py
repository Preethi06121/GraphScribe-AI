from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "localizerag_documents"
    chunk_size: int = 800
    chunk_overlap: int = 150
    temp_upload_dir: str = "./data/uploads"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    spacy_model: str = "en_core_web_sm"

    retrieval_top_k: int = 5
    retrieval_vector_weight: float = 0.7
    retrieval_graph_weight: float = 0.3
    retrieval_cache_size: int = 128

    llm_provider: str = "ollama"
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
