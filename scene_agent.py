"""
Agent 4: Scene Generator Agent
Converts the pedagogical plan into concrete Manim object/animation instructions.
Each scene is described in terms of Manim-native object types and animation calls.
"""

from __future__ import annotations

import logging

from backend.llm_client import llm_call
from backend.models import (
    ManimAnimation,
    ManimObject,
    PedagogyPlan,
    Scene,
    SceneInstruction,
    SceneInstructionSet,
)

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"Axes", "NumberLine", "Text", "MathTex", "Graph", "Arrow", "Dot", "Circle", "Rectangle"}
ALLOWED_ACTIONS = {"Create", "Write", "Transform", "FadeIn", "FadeOut", "GrowFromCenter", "ShowCreation"}

SYSTEM_PROMPT = """
You are a Manim animation engineer. Convert each pedagogical scene description
into precise Manim object and animation instructions.

ALLOWED object types: Axes, NumberLine, Text, MathTex, Arrow, Dot, Circle, Rectangle
ALLOWED animation actions: Create, Write, Transform, FadeIn, FadeOut, GrowFromCenter

Return a JSON object with EXACTLY this structure:
{
  "scene_instructions": [
    {
      "scene_id": 1,
      "objects": [
        {
          "obj_id": "axes_1",
          "obj_type": "Axes",
          "properties": {
            "x_range": [-3, 3, 1],
            "y_range": [-2, 2, 1],
            "axis_config": {"color": "BLUE"}
          }
        }
      ],
      "animations": [
        {
          "action": "Create",
          "target": "axes_1",
          "duration": 1.5,
          "kwargs": {}
        }
      ],
      "camera_actions": []
    }
  ]
}

Rules:
- obj_id must be a valid Python identifier (snake_case, unique within scene)
- animation "target" must exactly match an obj_id from the same scene
- animations are executed in ORDER — plan the sequence carefully
- For MathTex: properties must include "tex_string" key with valid LaTeX (escape backslashes: \\\\frac not \\frac)
- For Text: properties must include "text" key
- For Transform: include "source" and "target" in kwargs instead (source transforms into target)
- Keep each scene to 3–6 objects and 4–8 animations for clarity
- camera_actions can be empty list []
"""


def _sanitize_instruction(instr: SceneInstruction) -> SceneInstruction:
    """Clamp unknown object/action types to safe defaults to prevent Manim crashes."""
    sanitized_objects = []
    valid_ids = set()
    for obj in instr.objects:
        if obj.obj_type not in ALLOWED_TYPES:
            logger.warning("Unknown obj_type '%s' replaced with 'Text'", obj.obj_type)
            obj.obj_type = "Text"
            if "text" not in obj.properties:
                obj.properties["text"] = obj.obj_id
        valid_ids.add(obj.obj_id)
        sanitized_objects.append(obj)

    sanitized_anims = []
    for anim in instr.animations:
        if anim.action not in ALLOWED_ACTIONS:
            logger.warning("Unknown action '%s' replaced with 'FadeIn'", anim.action)
            anim.action = "FadeIn"
        # For non-Transform, verify target exists
        if anim.action != "Transform" and anim.target not in valid_ids:
            logger.warning("Animation target '%s' not found, skipping.", anim.target)
            continue
        sanitized_anims.append(anim)

    return SceneInstruction(
        scene_id=instr.scene_id,
        objects=sanitized_objects,
        animations=sanitized_anims,
        camera_actions=instr.camera_actions,
    )


def run(plan: PedagogyPlan) -> SceneInstructionSet:
    """Generate Manim scene instructions from the pedagogical plan."""

    scenes_text = "\n\n".join(
        f"Scene {s.scene_id}: {s.scene_title}\n"
        f"Learning goal: {s.learning_goal}\n"
        f"Visual metaphor: {s.visual_metaphor}\n"
        f"Equations to show: {s.equations_to_show}\n"
        f"Animation strategy: {s.animation_strategy}\n"
        f"Duration: ~{s.estimated_duration_seconds}s"
        for s in plan.scenes
    )

    result = llm_call(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Generate scene instructions for these scenes:\n\n{scenes_text}",
        response_model=SceneInstructionSet,
    )

    # Sanitize all instructions
    result.scene_instructions = [
        _sanitize_instruction(instr) for instr in result.scene_instructions
    ]

    logger.info("Scene generator produced %d scene instructions.", len(result.scene_instructions))
    return result
