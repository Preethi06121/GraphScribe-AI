from typing import Any

from pydantic import BaseModel, Field


class ContentGenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    audience: str = Field(..., min_length=1)
    country: str = Field(..., min_length=1)
    tone: str = Field(default="Professional")
    length: int = Field(default=1500, ge=200, le=10000)
    retrieval_strategy: str = Field(default="HYBRID")


class ArticleSection(BaseModel):
    heading: str
    content: str


class ArticleCitation(BaseModel):
    document_id: str
    document_name: str
    page: int | None = None
    source: str
    reference: str


class ArticleMetadata(BaseModel):
    retrieval_strategy: str = "HybridRAG"
    country: str
    audience: str
    tone: str
    topic: str
    word_count: int
    target_length: int


class VectorChunkExplainability(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    score: float
    excerpt: str


class GraphEntityExplainability(BaseModel):
    entity_name: str
    entity_type: str
    document_id: str
    document_name: str
    page_number: int
    score: float
    relationship_type: str | None = None
    connected_entity: str | None = None


class DocumentUsed(BaseModel):
    document_id: str
    document_name: str
    source_types: list[str] = Field(default_factory=list)


class Explainability(BaseModel):
    vector_chunks: list[VectorChunkExplainability]
    graph_entities: list[GraphEntityExplainability]
    reasoning: str
    documents_used: list[DocumentUsed]
    retrieval_strategy: str = "HybridRAG"
    vector_score_summary: float
    graph_score_summary: float
    chunk_count: int
    entity_count: int


class GeneratedArticleResponse(BaseModel):
    title: str
    summary: str
    sections: list[ArticleSection]
    conclusion: str
    citations: list[ArticleCitation]
    metadata: ArticleMetadata
    explainability: Explainability


class LLMArticlePayload(BaseModel):
    """Expected JSON structure from the LLM."""

    title: str
    summary: str
    sections: list[ArticleSection]
    conclusion: str
