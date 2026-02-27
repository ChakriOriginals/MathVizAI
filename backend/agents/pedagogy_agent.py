from __future__ import annotations
import logging
from backend.llm_client import llm_call
from backend.models import ConceptExtractionResult, PedagogyPlan
from backend.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert educational video producer in the style of 3Blue1Brown.
Design a sequence of animation scenes. Return a JSON object:
{
  "scenes": [
    {
      "scene_id": 1,
      "scene_title": "<short title>",
      "learning_goal": "<what viewer understands after this scene>",
      "visual_metaphor": "<concrete visual idea>",
      "equations_to_show": ["<LaTeX equation>"],
      "animation_strategy": "<how objects animate>",
      "estimated_duration_seconds": 40
    }
  ]
}
Rules: 3-5 scenes. Scene 1 MUST be an intuitive hook with equations_to_show: []. Final scene introduces the formal statement.
"""

def run(concepts: ConceptExtractionResult, difficulty_level: str = "undergraduate") -> PedagogyPlan:
    concept_text = "\n\n".join(
        f"Concept: {c.concept_name}\nExplanation: {c.intuitive_explanation}\nMath: {c.mathematical_form}\nSignificance: {c.why_it_matters}"
        for c in concepts.core_concepts
    )
    user_prompt = f"Difficulty level: {difficulty_level}\n\nConcept ordering: {', '.join(concepts.concept_ordering)}\n\nConcepts:\n{concept_text}"
    result = llm_call(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, response_model=PedagogyPlan)
    if len(result.scenes) > settings.max_scenes:
        result.scenes = result.scenes[:settings.max_scenes]
    for i, s in enumerate(result.scenes):
        s.scene_id = i + 1
    logger.info("Pedagogy plan: %s", [s.scene_title for s in result.scenes])
    return result