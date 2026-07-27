from pydantic import BaseModel, Field


class ProcessedQuery(BaseModel):
    original: str
    normalized: str
    keywords: list[str]
    semantic_query: str


class VectorHit(BaseModel):
    chunk: str
    score: float
    page: int
    document_id: str
    document_name: str = ""
    chunk_id: str = ""
    source: str = ""


class GraphHit(BaseModel):
    entity_name: str
    entity_type: str
    connected_entity: str | None = None
    connected_type: str | None = None
    relationship_type: str | None = None
    document_id: str
    document_name: str = ""
    page_number: int = 0
    score: float


class RankedItem(BaseModel):
    item_id: str
    item_type: str
    content: str
    document_id: str
    document_name: str = ""
    page: int = 0
    vector_score: float = 0.0
    graph_score: float = 0.0
    final_score: float
    source: str = ""
    metadata: dict = Field(default_factory=dict)


class ContextDocument(BaseModel):
    chunk: str
    score: float
    page: int
    document_id: str
    document_name: str = ""
    chunk_id: str = ""
    source: str = "vector"


class ContextGraphItem(BaseModel):
    entity_name: str
    entity_type: str
    connected_entity: str | None = None
    relationship_type: str | None = None
    document_id: str
    document_name: str = ""
    page_number: int = 0
    score: float
    source: str = "graph"


class Citation(BaseModel):
    document_id: str
    document_name: str
    page: int | None = None
    source: str
    reference: str


class HybridContextMetadata(BaseModel):
    query: str
    normalized_query: str
    keywords: list[str]
    vector_hits: int
    graph_hits: int
    vector_weight: float
    graph_weight: float


class HybridContext(BaseModel):
    documents: list[ContextDocument]
    graph: list[ContextGraphItem]
    citations: list[Citation]
    metadata: HybridContextMetadata


class RetrievalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)


class RetrievalSearchResponse(BaseModel):
    documents: list[ContextDocument]
    graph: list[ContextGraphItem]
    citations: list[Citation]
    metadata: HybridContextMetadata
