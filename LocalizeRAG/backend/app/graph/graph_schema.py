"""Neo4j graph schema definitions and constraint queries."""

ENTITY_LABEL = "Entity"

ENTITY_CONSTRAINT_QUERY = """
CREATE CONSTRAINT entity_unique IF NOT EXISTS
FOR (n:Entity)
REQUIRE (n.entity_name, n.entity_type, n.document_id, n.page_number) IS UNIQUE
"""

SPACY_ENTITY_TYPES = frozenset(
    {"PERSON", "ORG", "PRODUCT", "GPE", "EVENT", "WORK_OF_ART", "NORP"}
)

RULE_BASED_TECH_TERMS: dict[str, str] = {
    "llm": "TECH_TERM",
    "transformer": "TECH_TERM",
    "graphrag": "TECH_TERM",
    "rag": "TECH_TERM",
    "embeddings": "TECH_TERM",
    "vector database": "TECH_TERM",
    "knowledge graph": "TECH_TERM",
    "machine learning": "TECH_TERM",
    "deep learning": "TECH_TERM",
    "neural network": "TECH_TERM",
    "fine-tuning": "TECH_TERM",
    "fine tuning": "TECH_TERM",
    "lora": "TECH_TERM",
}
