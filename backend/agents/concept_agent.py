from __future__ import annotations
import logging
from backend.llm_client import llm_call
from backend.models import ConceptExtractionResult, ParsedContent
from backend.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert math educator. Given structured mathematical content, extract the 3-5 most important VISUALIZABLE concepts.
Return a JSON object with EXACTLY these keys:
{
  "core_concepts": [
    {
      "concept_name": "<short name>",
      "intuitive_explanation": "<1-2 sentence plain English explanation>",
      "mathematical_form": "<LaTeX expression>",
      "why_it_matters": "<1 sentence on significance>"
    }
  ],
  "concept_ordering": ["<concept_name_1>", "<concept_name_2>"]
}
Rules: concept_ordering lists all names in teaching order. Maximum 5 concepts.
"""

def run(parsed: ParsedContent, difficulty_level: str = "undergraduate") -> ConceptExtractionResult:
    user_prompt = (
        f"Difficulty level: {difficulty_level}\n\n"
        f"Main topic: {parsed.main_topic}\n\n"
        f"Definitions:\n" + "\n".join(f"- {d}" for d in parsed.definitions) + "\n\n"
        f"Key equations:\n" + "\n".join(f"- {e}" for e in parsed.key_equations) + "\n\n"
        f"Core claims:\n" + "\n".join(f"- {c}" for c in parsed.core_claims)
    )
    result = llm_call(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, response_model=ConceptExtractionResult)
    if len(result.core_concepts) > settings.max_concepts:
        result.core_concepts = result.core_concepts[:settings.max_concepts]
        result.concept_ordering = result.concept_ordering[:settings.max_concepts]
    logger.info("Concepts extracted: %s", [c.concept_name for c in result.core_concepts])
    return result