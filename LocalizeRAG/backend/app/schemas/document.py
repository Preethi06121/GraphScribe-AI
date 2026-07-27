from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    content: str
    document_id: str
    document_name: str
    page_number: int
    chunk_id: str
    source: str


class PageContent(BaseModel):
    page_number: int
    text: str


class IngestionResult(BaseModel):
    document_id: str
    document_name: str
    pages: int
    chunks: int


class DocumentUploadResponse(BaseModel):
    status: str = Field(default="success")
    document_id: str
    document: str
    pages: int
    chunks: int
