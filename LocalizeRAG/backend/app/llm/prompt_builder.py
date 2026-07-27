import json
import logging
from dataclasses import dataclass

from app.schemas.content import LLMArticlePayload
from app.schemas.retrieval import HybridContext

logger = logging.getLogger(__name__)

SUPPORTED_COUNTRIES = frozenset(
    {"India", "United States", "United Kingdom", "Japan", "Australia"}
)


@dataclass(frozen=True)
class LocalizationProfile:
    vocabulary: str
    examples: str
    writing_style: str
    units: str
    spelling: str
    educational_references: str
    cultural_context: str


LOCALIZATION_PROFILES: dict[str, LocalizationProfile] = {
    "India": LocalizationProfile(
        vocabulary="Use clear Indian English vocabulary accessible to students and professionals.",
        examples="Use India-relevant examples such as UPI, IITs, NEP, and domestic technology adoption.",
        writing_style="Structured, explanatory, and academically grounded with practical relevance.",
        units="Use metric units (km, kg, degrees Celsius, INR where relevant).",
        spelling="Use Indian English spelling conventions.",
        educational_references="Reference Indian curricula, competitive exams, and local industry context where appropriate.",
        cultural_context="Reflect India's multilingual, diverse, and rapidly digitizing society without stereotypes.",
    ),
    "United States": LocalizationProfile(
        vocabulary="Use American English vocabulary suited for technical and professional readers.",
        examples="Use US-relevant examples such as Silicon Valley, federal agencies, and major US tech firms.",
        writing_style="Direct, analytical, and outcome-oriented.",
        units="Use US customary and metric units where appropriate (miles, Fahrenheit, USD).",
        spelling="Use American English spelling conventions.",
        educational_references="Reference US university programs, industry certifications, and market trends where appropriate.",
        cultural_context="Reflect US innovation culture and enterprise adoption patterns.",
    ),
    "United Kingdom": LocalizationProfile(
        vocabulary="Use British English vocabulary with formal professional tone.",
        examples="Use UK-relevant examples such as NHS digital initiatives, UK research councils, and British industry.",
        writing_style="Balanced, precise, and evidence-led.",
        units="Use metric units with occasional imperial references where culturally common.",
        spelling="Use British English spelling conventions.",
        educational_references="Reference UK qualifications, research institutions, and policy context where appropriate.",
        cultural_context="Reflect UK public-sector and research-oriented technology discourse.",
    ),
    "Japan": LocalizationProfile(
        vocabulary="Use internationally clear English while acknowledging Japanese technology and business context.",
        examples="Use Japan-relevant examples such as robotics, manufacturing excellence, and domestic AI adoption.",
        writing_style="Formal, respectful, precise, and methodical.",
        units="Use metric units.",
        spelling="Use international English spelling with consistency.",
        educational_references="Reference Japanese research institutions, industry standards, and innovation ecosystems where appropriate.",
        cultural_context="Reflect Japan's emphasis on quality, process discipline, and long-term technology planning.",
    ),
    "Australia": LocalizationProfile(
        vocabulary="Use Australian English vocabulary with a professional educational tone.",
        examples="Use Australia-relevant examples such as national research infrastructure, local startups, and public-sector digital services.",
        writing_style="Clear, practical, and approachable while remaining professional.",
        units="Use metric units and AUD where relevant.",
        spelling="Use Australian English spelling conventions.",
        educational_references="Reference Australian universities, training pathways, and regional industry context where appropriate.",
        cultural_context="Reflect Australia's pragmatic approach to technology adoption and regional diversity.",
    ),
}


class PromptBuilder:
    """Builds structured prompts for long-form article generation."""

    def build_article_prompt(
        self,
        topic: str,
        audience: str,
        country: str,
        tone: str,
        length: int,
        context: HybridContext,
    ) -> str:
        localization = self._resolve_localization(country)
        vector_context = self._format_vector_context(context)
        graph_context = self._format_graph_context(context)
        output_schema = self._output_schema()

        sections = [
            self._section("System Role", self._system_role()),
            self._section("Task", self._task_section(topic, length)),
            self._section("Audience", audience),
            self._section("Country", country),
            self._section("Tone", tone),
            self._section("Target Length", f"Approximately {length} words."),
            self._section("Retrieved Vector Context", vector_context),
            self._section("Knowledge Graph Context", graph_context),
            self._section("Writing Instructions", self._writing_instructions(localization)),
            self._section("Citation Instructions", self._citation_instructions()),
            self._section("Output Schema", output_schema),
        ]

        prompt = "\n\n".join(sections)
        logger.info("Built article prompt for topic='%s' country='%s'", topic, country)
        return prompt

    def _resolve_localization(self, country: str) -> LocalizationProfile:
        if country not in SUPPORTED_COUNTRIES:
            raise ValueError(
                f"Unsupported country '{country}'. "
                f"Supported countries: {', '.join(sorted(SUPPORTED_COUNTRIES))}"
            )
        return LOCALIZATION_PROFILES[country]

    @staticmethod
    def _section(title: str, body: str) -> str:
        return f"## {title}\n{body.strip()}"

    @staticmethod
    def _system_role() -> str:
        return (
            "You are an expert research writer and localization specialist. "
            "You produce accurate, professional long-form articles grounded in the provided context."
        )

    @staticmethod
    def _task_section(topic: str, length: int) -> str:
        return (
            f"Generate a professional long-form article about '{topic}'. "
            f"Target length: approximately {length} words. "
            "The article must be informative, well-structured, and suitable for publication."
        )

    @staticmethod
    def _writing_instructions(localization: LocalizationProfile) -> str:
        return (
            f"Vocabulary: {localization.vocabulary}\n"
            f"Examples: {localization.examples}\n"
            f"Writing Style: {localization.writing_style}\n"
            f"Units: {localization.units}\n"
            f"Spelling: {localization.spelling}\n"
            f"Educational References: {localization.educational_references}\n"
            f"Cultural Context: {localization.cultural_context}\n"
            "Do not change factual information from the retrieved context. "
            "Do not invent citations. If context is insufficient, state limitations clearly."
        )

    @staticmethod
    def _citation_instructions() -> str:
        return (
            "Ground claims in the retrieved vector and graph context. "
            "Reference source documents implicitly within the prose where appropriate. "
            "Do not fabricate document names, page numbers, or entities."
        )

    @staticmethod
    def _format_vector_context(context: HybridContext) -> str:
        if not context.documents:
            return "No vector context retrieved."

        blocks: list[str] = []
        for index, doc in enumerate(context.documents, start=1):
            blocks.append(
                f"[Vector {index}] document_id={doc.document_id}; "
                f"document_name={doc.document_name}; page={doc.page}; score={doc.score:.3f}\n"
                f"{doc.chunk}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _format_graph_context(context: HybridContext) -> str:
        if not context.graph:
            return "No knowledge graph context retrieved."

        blocks: list[str] = []
        for index, item in enumerate(context.graph, start=1):
            relation = ""
            if item.relationship_type and item.connected_entity:
                relation = f" -[{item.relationship_type}]-> {item.connected_entity}"
            blocks.append(
                f"[Graph {index}] {item.entity_name} ({item.entity_type}){relation}; "
                f"document_id={item.document_id}; score={item.score:.3f}"
            )
        return "\n".join(blocks)

    @staticmethod
    def _output_schema() -> str:
        schema = {
            "title": "string",
            "summary": "string",
            "sections": [{"heading": "string", "content": "string"}],
            "conclusion": "string",
        }
        return (
            "Return valid JSON only. Do not include markdown fences or commentary.\n"
            f"Schema:\n{json.dumps(schema, indent=2)}"
        )
