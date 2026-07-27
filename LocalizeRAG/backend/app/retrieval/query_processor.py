import logging
import re

from app.schemas.retrieval import ProcessedQuery

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "with",
        "by",
        "from",
        "as",
        "into",
        "about",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "how",
        "why",
        "when",
        "where",
        "can",
        "could",
        "should",
        "would",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "it",
        "its",
        "their",
        "them",
        "they",
        "we",
        "you",
        "your",
        "our",
        "i",
        "me",
        "my",
    }
)


class QueryProcessor:
    """Normalizes queries and extracts keywords for hybrid retrieval."""

    def process(self, query: str) -> ProcessedQuery:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        original = query
        normalized = self._normalize(query)
        keywords = self._extract_keywords(normalized)
        semantic_query = normalized

        logger.info(
            "Processed query: keywords=%d semantic_len=%d",
            len(keywords),
            len(semantic_query),
        )
        return ProcessedQuery(
            original=original,
            normalized=normalized,
            keywords=keywords,
            semantic_query=semantic_query,
        )

    @staticmethod
    def _normalize(query: str) -> str:
        cleaned = query.strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"[^\w\s\-./]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _extract_keywords(normalized: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9][a-z0-9\-./]*", normalized)
        keywords: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in STOPWORDS or len(token) < 2:
                continue
            if token not in seen:
                seen.add(token)
                keywords.append(token)
        return keywords
