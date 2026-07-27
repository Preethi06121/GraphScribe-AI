import logging
import re
from functools import lru_cache

import spacy

from app.graph.graph_schema import RULE_BASED_TECH_TERMS, SPACY_ENTITY_TYPES
from app.schemas.graph import GraphEntity

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts named entities using spaCy and rule-based technical term matching."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._model_name = model_name
        self._nlp: spacy.Language | None = None

    def _get_nlp(self) -> spacy.Language:
        if self._nlp is None:
            logger.info("Loading spaCy model: %s", self._model_name)
            self._nlp = spacy.load(self._model_name)
            logger.info("spaCy model loaded: %s", self._model_name)
        return self._nlp

    def extract_from_text(
        self,
        text: str,
        document_id: str,
        document_name: str,
        page_number: int,
    ) -> list[GraphEntity]:
        nlp = self._get_nlp()
        doc = nlp(text)

        entities: dict[tuple[str, str], GraphEntity] = {}

        for ent in doc.ents:
            if ent.label_ not in SPACY_ENTITY_TYPES:
                continue
            key = (ent.text.strip(), ent.label_)
            if key not in entities:
                entities[key] = GraphEntity(
                    entity_name=ent.text.strip(),
                    entity_type=ent.label_,
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                )

        for term, entity_type in RULE_BASED_TECH_TERMS.items():
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                key = (matched_text, entity_type)
                if key not in entities:
                    entities[key] = GraphEntity(
                        entity_name=matched_text,
                        entity_type=entity_type,
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_number,
                    )

        result = list(entities.values())
        logger.debug(
            "Extracted %d entit(ies) from page %d of %s",
            len(result),
            page_number,
            document_name,
        )
        return result


    def get_nlp_doc(self, text: str):
        return self._get_nlp()(text)


@lru_cache
def get_entity_extractor(model_name: str = "en_core_web_sm") -> EntityExtractor:
    return EntityExtractor(model_name=model_name)
