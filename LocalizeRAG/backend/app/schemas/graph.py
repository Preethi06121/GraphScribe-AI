from pydantic import BaseModel, Field


class GraphEntity(BaseModel):
    entity_name: str
    entity_type: str
    document_id: str
    document_name: str
    page_number: int


class GraphRelationship(BaseModel):
    source_entity: str
    source_type: str
    relationship_type: str
    target_entity: str
    target_type: str
    document_id: str
    source_document: str
    page_number: int


class GraphStatistics(BaseModel):
    nodes: int
    relationships: int
    entity_types: list[str]
    documents: int


class DocumentGraphNode(BaseModel):
    entity_name: str
    entity_type: str
    document_id: str
    document_name: str
    page_number: int


class DocumentGraphRelationship(BaseModel):
    source_entity: str
    source_type: str
    relationship_type: str
    target_entity: str
    target_type: str
    document_id: str
    source_document: str
    page_number: int


class DocumentGraphStatistics(BaseModel):
    nodes: int
    relationships: int
    entity_types: list[str]


class DocumentGraphResponse(BaseModel):
    document_id: str
    nodes: list[DocumentGraphNode]
    relationships: list[DocumentGraphRelationship]
    statistics: DocumentGraphStatistics
