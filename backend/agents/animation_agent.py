from __future__ import annotations
import ast
import logging
import re
from backend.llm_client import llm_call
from backend.models import AnimationCode, PedagogyPlan, SceneInstructionSet
from backend.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Generate a complete, runnable Manim (Community Edition v0.18+) Python script.

RULES:
1. Use ONLY: from manim import *
2. ONE class extending Scene named MathVizScene
3. Implement construct(self) method
4. Use self.play(...) for animations, self.wait(n) between scenes
5. Every variable in self.play() MUST be defined before that line
6. Use Write for Text/MathTex, Create for shapes/axes
7. Use FadeOut(*self.mobjects) to clear between scenes
8. Escape all LaTeX backslashes (\\frac, \\sum, \\int)
9. Keep total code under 400 lines
10. Add # === Scene N: Title === comments

Return JSON:
{
  "manim_class_name": "MathVizScene",
  "python_code": "<complete Python script>"
}
"""

def _fix_common_issues(code: str) -> str:
    import ast
    import logging
    logger = logging.getLogger(__name__)

    # 1. Strip markdown fences
    code = re.sub(r"```python\s*", "", code)
    code = re.sub(r"```\s*", "", code)

    # 2. Ensure manim import exists
    if "from manim import" not in code:
        code = "from manim import *\n\n" + code

    # 3. Fix ShowCreation -> Create
    code = code.replace("ShowCreation(", "Create(")
    # Fix: undefined Manim colors -> valid alternatives
    code = code.replace('CYAN', 'TEAL')
    code = code.replace('MAGENTA', 'PINK')
    code = code.replace('BROWN', 'DARK_BROWN')
    code = code.replace('LIGHT_BLUE', 'BLUE_B')
    code = code.replace('DARK_BLUE', 'DARK_BLUE')
    code = code.replace('LIGHT_GREEN', 'GREEN_B')
    code = code.replace('LIGHT_RED', 'RED_B')
    code = code.replace('DARK_RED', 'MAROON')

    # 4. Fix duration= -> run_time= everywhere
    code = re.sub(r',\s*duration=([0-9.]+)', r', run_time=\1', code)

    # 5. Replace ALL MathTex(...) and Tex(...) with Text(...)
    #    Handle multiline and complex arguments
    code = re.sub(r'\bMathTex\b', 'Text', code)
    code = re.sub(r'\bTex\b(?!\w)', 'Text', code)

    # 6. Fix duplicate keyword arguments (e.g. font_size=36, font_size=36)
    #    Use line-by-line regex to remove duplicate kwargs
    def remove_duplicate_kwargs(line):
        # Find all kwarg=value pairs and deduplicate
        seen = {}
        def replace_kwarg(m):
            key = m.group(1)
            if key in seen:
                return ''  # Remove duplicate
            seen[key] = True
            return m.group(0)
        # Match keyword=value patterns inside function calls
        result = re.sub(r'\b([a-zA-Z_]\w*)\s*=\s*[^,)]+', replace_kwarg, line)
        # Clean up any resulting double commas or trailing commas before )
        result = re.sub(r',\s*,', ',', result)
        result = re.sub(r',\s*\)', ')', result)
        return result

    code = '\n'.join(remove_duplicate_kwargs(line) for line in code.split('\n'))

    # 7. Fix 2D coords -> 3D (Manim requires [x, y, 0])
    def fix_2d(m):
        prefix, x, y = m.group(1), m.group(2).strip(), m.group(3).strip()
        return f'{prefix}[{x}, {y}, 0]'

    for kw in ['point', 'start', 'end', 'arc_center']:
        code = re.sub(rf'(\b{kw}\s*=\s*)\[([^,\[\]]+),\s*([^,\[\]]+)\]', fix_2d, code)
    code = re.sub(r'(\.move_to\s*\()\[([^,\[\]]+),\s*([^,\[\]]+)\]',
                  lambda m: f'{m.group(1)}[{m.group(2).strip()}, {m.group(3).strip()}, 0]', code)
    code = re.sub(r'(\.shift\s*\()\[([^,\[\]]+),\s*([^,\[\]]+)\]',
                  lambda m: f'{m.group(1)}[{m.group(2).strip()}, {m.group(3).strip()}, 0]', code)

    # 8. Syntax check — fallback to safe scene if still broken
    try:
        ast.parse(code)
    except SyntaxError as e:
        logger.error(f"Syntax error after all fixes: {e}")
        code = '''from manim import *

class MathVizScene(Scene):
    def construct(self):
        title = Text("Math Visualization", font_size=48)
        subtitle = Text("Animation generation encountered an error", font_size=28, color=YELLOW)
        subtitle.next_to(title, DOWN)
        self.play(Write(title, run_time=1.5))
        self.play(FadeIn(subtitle, run_time=1.0))
        self.wait(2)
'''
    return code

def _syntax_check(code: str):
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)

def run(scene_instructions: SceneInstructionSet, plan: PedagogyPlan) -> AnimationCode:
    context_parts = []
    for instr in scene_instructions.scene_instructions:
        plan_scene = next((s for s in plan.scenes if s.scene_id == instr.scene_id), None)
        title = plan_scene.scene_title if plan_scene else f"Scene {instr.scene_id}"
        goal = plan_scene.learning_goal if plan_scene else ""
        objects_desc = "\n".join(f"  - {o.obj_id} ({o.obj_type}): {o.properties}" for o in instr.objects)
        anims_desc = "\n".join(f"  - {a.action}({a.target}, duration={a.duration})" for a in instr.animations)
        context_parts.append(f"Scene {instr.scene_id}: {title}\nGoal: {goal}\nObjects:\n{objects_desc}\nAnimations:\n{anims_desc}")
    user_prompt = "Generate a Manim script for these scenes:\n\n" + "\n\n".join(context_parts)
    result = llm_call(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, response_model=AnimationCode, max_retries=2)
    result.python_code = _fix_common_issues(result.python_code)
    valid, err = _syntax_check(result.python_code)
    if not valid:
        raise ValueError(f"Generated Manim code has a syntax error: {err}")
    lines = len(result.python_code.strip().splitlines())
    logger.info("Animation code generated: class=%s, lines=%d", result.manim_class_name, lines)
    return result