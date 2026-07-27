import logging
import re

from app.schemas.graph import GraphEntity, GraphRelationship

logger = logging.getLogger(__name__)

VERB_TO_RELATIONSHIP = {
    "develop": "DEVELOPED",
    "developed": "DEVELOPED",
    "develops": "DEVELOPED",
    "create": "CREATED",
    "created": "CREATED",
    "creates": "CREATED",
    "produce": "PRODUCES",
    "produced": "PRODUCES",
    "produces": "PRODUCES",
    "build": "BUILT",
    "built": "BUILT",
    "builds": "BUILT",
    "found": "FOUNDED",
    "founded": "FOUNDED",
    "founds": "FOUNDED",
    "acquire": "ACQUIRED",
    "acquired": "ACQUIRED",
    "acquires": "ACQUIRED",
    "launch": "LAUNCHED",
    "launched": "LAUNCHED",
    "launches": "LAUNCHED",
    "use": "USES",
    "used": "USES",
    "uses": "USES",
    "own": "OWNS",
    "owned": "OWNS",
    "owns": "OWNS",
}


class RelationshipExtractor:
    """Extracts subject-verb-object relationships and co-occurrence links."""

    def extract(
        self,
        text: str,
        entities: list[GraphEntity],
        document_id: str,
        document_name: str,
        page_number: int,
        nlp_doc,
    ) -> list[GraphRelationship]:
        relationships: list[GraphRelationship] = []
        seen: set[tuple[str, str, str, str]] = set()

        entity_lookup = {ent.entity_name.lower(): ent for ent in entities}

        for sent in nlp_doc.sents:
            subject = None
            verb = None
            obj = None

            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and token.ent_type_:
                    subject = token
                elif token.pos_ == "VERB":
                    verb = token
                elif token.dep_ in ("dobj", "attr", "pobj") and token.ent_type_:
                    obj = token

            if subject and verb and obj:
                rel_type = VERB_TO_RELATIONSHIP.get(
                    verb.lemma_.lower(),
                    verb.lemma_.upper(),
                )
                source_ent = entity_lookup.get(subject.text.lower())
                target_ent = entity_lookup.get(obj.text.lower())
                if source_ent and target_ent:
                    key = (
                        source_ent.entity_name,
                        rel_type,
                        target_ent.entity_name,
                        document_id,
                    )
                    if key not in seen:
                        seen.add(key)
                        relationships.append(
                            GraphRelationship(
                                source_entity=source_ent.entity_name,
                                source_type=source_ent.entity_type,
                                relationship_type=rel_type,
                                target_entity=target_ent.entity_name,
                                target_type=target_ent.entity_type,
                                document_id=document_id,
                                source_document=document_name,
                                page_number=page_number,
                            )
                        )

        if not relationships and len(entities) >= 2:
            for i, source in enumerate(entities):
                for target in entities[i + 1 :]:
                    key = (
                        source.entity_name,
                        "RELATED_TO",
                        target.entity_name,
                        document_id,
                    )
                    if key not in seen:
                        seen.add(key)
                        relationships.append(
                            GraphRelationship(
                                source_entity=source.entity_name,
                                source_type=source.entity_type,
                                relationship_type="RELATED_TO",
                                target_entity=target.entity_name,
                                target_type=target.entity_type,
                                document_id=document_id,
                                source_document=document_name,
                                page_number=page_number,
                            )
                        )

        self._extract_pattern_relationships(
            text,
            entities,
            document_id,
            document_name,
            page_number,
            relationships,
            seen,
        )

        logger.debug(
            "Extracted %d relationship(s) from page %d of %s",
            len(relationships),
            page_number,
            document_name,
        )
        return relationships

    def _extract_pattern_relationships(
        self,
        text: str,
        entities: list[GraphEntity],
        document_id: str,
        document_name: str,
        page_number: int,
        relationships: list[GraphRelationship],
        seen: set[tuple[str, str, str, str]],
    ) -> None:
        entity_names = {ent.entity_name for ent in entities}
        pattern = re.compile(
            r"(?P<subject>[A-Z][\w\s&.-]+?)\s+"
            r"(?P<verb>developed|develops|produced|produces|created|creates|built|builds|founded|founds|launched|launches|acquired|acquires)\s+"
            r"(?P<object>[A-Z][\w\s&.-]+)",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            subject_text = match.group("subject").strip()
            verb_text = match.group("verb").strip().lower()
            object_text = match.group("object").strip().rstrip(".")

            source_ent = self._find_entity(subject_text, entities, entity_names)
            target_ent = self._find_entity(object_text, entities, entity_names)
            if not source_ent or not target_ent:
                continue

            rel_type = VERB_TO_RELATIONSHIP.get(verb_text, verb_text.upper())
            key = (
                source_ent.entity_name,
                rel_type,
                target_ent.entity_name,
                document_id,
            )
            if key in seen:
                continue
            seen.add(key)
            relationships.append(
                GraphRelationship(
                    source_entity=source_ent.entity_name,
                    source_type=source_ent.entity_type,
                    relationship_type=rel_type,
                    target_entity=target_ent.entity_name,
                    target_type=target_ent.entity_type,
                    document_id=document_id,
                    source_document=document_name,
                    page_number=page_number,
                )
            )

    @staticmethod
    def _find_entity(
        text: str,
        entities: list[GraphEntity],
        entity_names: set[str],
    ) -> GraphEntity | None:
        if text in entity_names:
            for ent in entities:
                if ent.entity_name == text:
                    return ent
        text_lower = text.lower()
        for ent in entities:
            if ent.entity_name.lower() == text_lower:
                return ent
            if ent.entity_name.lower() in text_lower or text_lower in ent.entity_name.lower():
                return ent
        return None
