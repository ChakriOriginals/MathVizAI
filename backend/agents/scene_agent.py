from __future__ import annotations
import logging
from backend.llm_client import llm_call
from backend.models import ManimAnimation, ManimObject, PedagogyPlan, SceneInstruction, SceneInstructionSet

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"Axes", "NumberLine", "Text", "MathTex", "Graph", "Arrow", "Dot", "Circle", "Rectangle"}
ALLOWED_ACTIONS = {"Create", "Write", "Transform", "FadeIn", "FadeOut", "GrowFromCenter", "ShowCreation"}

SYSTEM_PROMPT = """
Convert pedagogical scenes into Manim animation instructions.
ALLOWED object types: Axes, NumberLine, Text, MathTex, Arrow, Dot, Circle, Rectangle
ALLOWED animation actions: Create, Write, Transform, FadeIn, FadeOut, GrowFromCenter

Return ONLY this JSON structure, keep it concise:
{
  "scene_instructions": [
    {
      "scene_id": 1,
      "objects": [
        {"obj_id": "title_1", "obj_type": "Text", "properties": {"text": "Hello"}}
      ],
      "animations": [
        {"action": "Write", "target": "title_1", "duration": 1.5, "kwargs": {}}
      ],
      "camera_actions": []
    }
  ]
}
Rules:
- Maximum 4 objects per scene
- Maximum 4 animations per scene
- obj_id must be unique snake_case within each scene
- animation target must exactly match an obj_id
- Keep property values short strings only
"""

def _sanitize_instruction(instr: SceneInstruction) -> SceneInstruction:
    valid_ids = set()
    sanitized_objects = []
    for obj in instr.objects:
        if obj.obj_type not in ALLOWED_TYPES:
            obj.obj_type = "Text"
            if "text" not in obj.properties:
                obj.properties["text"] = obj.obj_id
        valid_ids.add(obj.obj_id)
        sanitized_objects.append(obj)
    sanitized_anims = []
    for anim in instr.animations:
        if anim.action not in ALLOWED_ACTIONS:
            anim.action = "FadeIn"
        if anim.action != "Transform" and anim.target not in valid_ids:
            logger.warning("Animation target '%s' not found, skipping.", anim.target)
            continue
        sanitized_anims.append(anim)
    return SceneInstruction(scene_id=instr.scene_id, objects=sanitized_objects, animations=sanitized_anims, camera_actions=instr.camera_actions)

def run(plan: PedagogyPlan) -> SceneInstructionSet:
    scenes_text = "\n\n".join(
        f"Scene {s.scene_id}: {s.scene_title}\nGoal: {s.learning_goal}\nMetaphor: {s.visual_metaphor}\nEquations: {s.equations_to_show}\nStrategy: {s.animation_strategy}"
        for s in plan.scenes
    )
    result = llm_call(system_prompt=SYSTEM_PROMPT, user_prompt=f"Generate scene instructions:\n\n{scenes_text}", response_model=SceneInstructionSet)
    result.scene_instructions = [_sanitize_instruction(i) for i in result.scene_instructions]
    logger.info("Scene instructions generated: %d scenes", len(result.scene_instructions))
    return result