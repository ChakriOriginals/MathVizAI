"""
Agent 3: Pedagogical Planner Agent
Designs a 3Blue1Brown-style learning sequence from extracted concepts.
Produces a structured sequence of scenes with visual strategies.
"""

from __future__ import annotations

import logging

from backend.llm_client import llm_call
from backend.models import ConceptExtractionResult, PedagogyPlan
from backend.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert educational video producer in the style of 3Blue1Brown.
You create pedagogically optimal scene sequences that:
1. Start with a compelling intuitive hook (NO equations at first)
2. Build understanding gradually with visual metaphors
3. Introduce formalism only after intuition is established
4. End with the formal mathematical statement

Given extracted concepts, design a sequence of animation scenes.

Return a JSON object with EXACTLY this structure:
{
  "scenes": [
    {
      "scene_id": 1,
      "scene_title": "<short title>",
      "learning_goal": "<what the viewer will understand after this scene>",
      "visual_metaphor": "<concrete visual or geometric idea to show>",
      "equations_to_show": ["<LaTeX equation 1>", ...],
      "animation_strategy": "<description of how objects animate: e.g., 'NumberLine grows, then dots appear representing samples'>",
      "estimated_duration_seconds": 40
    }
  ]
}

Rules:
- MUST have between 3 and 5 scenes
- Scene 1 MUST be an intuitive hook with NO equations (equations_to_show: [])
- Final scene MUST introduce the formal mathematical statement
- Each scene should be 30–60 seconds
- visual_metaphor must be something Manim can animate (number lines, graphs, transformations, etc.)
- animation_strategy must be specific enough to generate Manim code
"""


def run(
    concepts: ConceptExtractionResult,
    difficulty_level: str = "undergraduate",
) -> PedagogyPlan:
    """Create a pedagogically optimized scene plan from extracted concepts."""

    concept_text = "\n\n".join(
        f"Concept: {c.concept_name}\n"
        f"Explanation: {c.intuitive_explanation}\n"
        f"Math: {c.mathematical_form}\n"
        f"Significance: {c.why_it_matters}"
        for c in concepts.core_concepts
    )

    user_prompt = (
        f"Difficulty level: {difficulty_level}\n\n"
        f"Concept ordering: {', '.join(concepts.concept_ordering)}\n\n"
        f"Concepts:\n{concept_text}"
    )

    result = llm_call(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=PedagogyPlan,
    )

    # Enforce scene count
    if len(result.scenes) > settings.max_scenes:
        result.scenes = result.scenes[: settings.max_scenes]
        # Re-number
        for i, s in enumerate(result.scenes):
            s.scene_id = i + 1

    # Ensure scene IDs are sequential
    for i, s in enumerate(result.scenes):
        s.scene_id = i + 1

    logger.info(
        "Pedagogy plan created. Scenes: %s",
        [s.scene_title for s in result.scenes],
    )
    return result
