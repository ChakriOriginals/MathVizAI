"""
Agent 5: Animation Code Generator
Converts structured scene instructions into a single runnable Manim Python script.
Applies post-processing to fix common LLM mistakes before returning code.
"""

from __future__ import annotations

import ast
import logging
import re

from backend.llm_client import llm_call
from backend.models import AnimationCode, PedagogyPlan, SceneInstructionSet
from backend.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert Manim (Community Edition v0.18+) developer.
Generate a complete, runnable Manim Python script from scene instructions.

IMPORTANT RULES:
1. Use ONLY: `from manim import *`
2. Create ONE class that extends Scene: class MathVizScene(Scene)
3. Implement the `construct(self)` method
4. Use `self.play(...)` for animations, `self.wait(n)` between scenes
5. Use `self.add(...)` only for non-animated objects
6. Every variable referenced in self.play() MUST be defined before that line
7. Use `Write` for Text/MathTex, `Create` for shapes/axes
8. Use `FadeOut(*self.mobjects)` to clear between scenes
9. For MathTex: escape all LaTeX backslashes (\\frac, \\sum, \\int etc.)
10. Keep total code under 400 lines
11. Add a 1–2 second wait after each animation group
12. Each scene section should have a comment: # === Scene N: Title ===

VALID MANIM PATTERNS:
```python
from manim import *

class MathVizScene(Scene):
    def construct(self):
        # === Scene 1: Hook ===
        title = Text("Central Limit Theorem", font_size=48)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))
        
        # === Scene 2: Setup ===
        axes = Axes(x_range=[-3,3,1], y_range=[0,1,0.5])
        self.play(Create(axes))
        self.wait(1)
```

Return a JSON object with EXACTLY:
{
  "manim_class_name": "MathVizScene",
  "python_code": "<complete Python script as a single string>"
}
"""


def _fix_common_issues(code: str) -> str:
    """Apply deterministic post-processing fixes for common LLM Manim mistakes."""

    # Ensure correct import
    if "from manim import" not in code:
        code = "from manim import *\n\n" + code

    # Remove any ```python fences that leaked in
    code = re.sub(r"```python\s*", "", code)
    code = re.sub(r"```\s*", "", code)

    # Fix a common mistake: ShowCreation → Create (ShowCreation removed in newer Manim)
    code = code.replace("ShowCreation(", "Create(")

    # Fix double-escaped backslashes that become quadruple in generated code
    # (LLMs sometimes over-escape)
    # We normalize \\\\frac → \\frac inside MathTex string arguments
    # This is a best-effort heuristic
    code = re.sub(r'\\\\\\\\', r'\\\\', code)

    return code


def _syntax_check(code: str) -> tuple[bool, str]:
    """Return (is_valid, error_message)."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


def _count_lines(code: str) -> int:
    return len(code.strip().splitlines())


def run(
    scene_instructions: SceneInstructionSet,
    plan: PedagogyPlan,
) -> AnimationCode:
    """Generate a complete Manim Python script from scene instructions."""

    # Build a rich context combining pedagogy plan and scene instructions
    context_parts = []
    for scene_instr in scene_instructions.scene_instructions:
        # Find matching plan scene for title/goal context
        plan_scene = next(
            (s for s in plan.scenes if s.scene_id == scene_instr.scene_id), None
        )
        title = plan_scene.scene_title if plan_scene else f"Scene {scene_instr.scene_id}"
        goal = plan_scene.learning_goal if plan_scene else ""

        objects_desc = "\n".join(
            f"  - {o.obj_id} ({o.obj_type}): {o.properties}"
            for o in scene_instr.objects
        )
        anims_desc = "\n".join(
            f"  - {a.action}({a.target}, duration={a.duration})"
            for a in scene_instr.animations
        )
        context_parts.append(
            f"Scene {scene_instr.scene_id}: {title}\n"
            f"Goal: {goal}\n"
            f"Objects:\n{objects_desc}\n"
            f"Animations:\n{anims_desc}"
        )

    user_prompt = "Generate a Manim script for these scenes:\n\n" + "\n\n".join(context_parts)

    result = llm_call(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=AnimationCode,
        max_retries=2,
    )

    # Post-process
    result.python_code = _fix_common_issues(result.python_code)

    # Syntax validation
    valid, err = _syntax_check(result.python_code)
    if not valid:
        raise ValueError(
            f"Generated Manim code has a syntax error: {err}\n\n"
            "Please retry — the LLM produced invalid Python."
        )

    # Line count guard
    lines = _count_lines(result.python_code)
    if lines > settings.max_manim_lines:
        logger.warning(
            "Generated code is %d lines (limit %d). It may be truncated.",
            lines, settings.max_manim_lines
        )

    logger.info(
        "Animation code generated: class=%s, lines=%d",
        result.manim_class_name, lines
    )
    return result
