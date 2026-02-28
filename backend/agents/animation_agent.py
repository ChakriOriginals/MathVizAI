from __future__ import annotations
import ast
import logging
import re
from backend.llm_client import llm_call
from backend.models import AnimationCode, PedagogyPlan, SceneInstructionSet
from backend.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert Manim animator creating 3Blue1Brown-style math education videos.

CRITICAL RULES:
1. Use MathTex() for ALL mathematical expressions, equations, and formulas — never Text() for math
2. Use Text() ONLY for plain English labels and titles
3. Every MathTex string must be valid LaTeX. Use raw strings: MathTex(r"a^2 + b^2 = c^2")
4. ALL coordinates must be 3D: use np.array([x, y, 0]) or [x, y, 0], never [x, y]
5. Use run_time= not duration= for animation timing
6. Keep all objects within screen bounds: x in [-6, 6], y in [-3.5, 3.5]
7. Clear screen between major sections using self.play(FadeOut(*self.mobjects))
8. ONLY animate concepts from the provided scene instructions — do not invent new examples

GOOD MATH PATTERNS:
    eq = MathTex(r"a^2 + b^2 = c^2", font_size=42)
    label = Text("Pythagorean Theorem", font_size=36)
    axes = Axes(x_range=[-3, 3, 1], y_range=[-2, 2, 1])

BAD PATTERNS (never do these):
    Text(r"a^2 + b^2 = c^2")  # Math must use MathTex
    Dot(point=[0, 0])  # Missing z coordinate
    self.play(Write(obj, duration=1.5))  # Wrong kwarg name

TIMING: Each scene should have self.wait() calls between animations. Total video ~3-4 minutes.

Generate a single Python class MathVizScene(Scene) with all scenes in construct().
Return ONLY the Python code, no explanation.
"""

RESPONSE_FORMAT = """
You MUST respond with JSON in exactly this format:
{
  "manim_class_name": "MathVizScene",
  "python_code": "from manim import *\\n\\nclass MathVizScene(Scene):\\n    def construct(self):\\n        ..."
}
Use "python_code" and "manim_class_name" keys only. Never use a "code" key.
"""

MATHTEX_SAFE_WRAPPER = '''
def safe_tex(latex_str, **kwargs):
    try:
        return MathTex(latex_str, **kwargs)
    except Exception:
        clean = latex_str.replace("\\\\\\\\", "").replace("{", "").replace("}", "")
        return Text(clean[:60], font_size=kwargs.get("font_size", 36))

'''


def _fix_common_issues(code: str) -> str:

    # 1. Strip markdown fences
    code = re.sub(r"```python\s*", "", code)
    code = re.sub(r"```\s*", "", code)

    # 2. Ensure manim import
    if "from manim import" not in code:
        code = "from manim import *\n\n" + code

    # 3. Fix deprecated API
    code = code.replace("ShowCreation(", "Create(")

    # 4. Fix duration= -> run_time=
    code = re.sub(r',\s*duration=([0-9.]+)', r', run_time=\1', code)

    # 5. Fix undefined colors
    color_fixes = {
        'CYAN': 'TEAL',
        'MAGENTA': 'PINK',
        'BROWN': 'DARK_BROWN',
        'LIGHT_BLUE': 'BLUE_B',
        'LIGHT_GREEN': 'GREEN_B',
        'LIGHT_RED': 'RED_B',
        'DARK_RED': 'MAROON',
        'LIME': 'GREEN_A',
        'NAVY': 'DARK_BLUE',
        'VIOLET': 'PURPLE',
        'SALMON': 'RED_B',
        'INDIGO': 'PURPLE_B',
    }
    for bad, good in color_fixes.items():
        code = re.sub(rf'\b{bad}\b(?!["\'])', good, code)

    # 6. Fix duplicate keyword arguments on same line
    def dedup_kwargs(line):
        keys_found = []
        def replacer(m):
            key = m.group(1)
            if key in ('True', 'False', 'None') or key[0].isupper():
                return m.group(0)
            if key in keys_found:
                return ''
            keys_found.append(key)
            return m.group(0)
        line = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,=()]+)(?=\s*,|\s*\))', replacer, line)
        line = re.sub(r',\s*,', ',', line)
        line = re.sub(r',\s*\)', ')', line)
        return line

    code = '\n'.join(dedup_kwargs(line) for line in code.split('\n'))

    # 7. Fix 2D coordinates -> 3D
    def fix_2d(m):
        prefix, x, y = m.group(1), m.group(2).strip(), m.group(3).strip()
        return f'{prefix}[{x}, {y}, 0]'

    for kw in ['point', 'start', 'end', 'arc_center']:
        code = re.sub(rf'(\b{kw}\s*=\s*)\[([^,\[\]]+),\s*([^,\[\]]+)\]', fix_2d, code)
    code = re.sub(r'(\.move_to\s*\()\[([^,\[\]]+),\s*([^,\[\]]+)\]',
                  lambda m: f'{m.group(1)}[{m.group(2).strip()}, {m.group(3).strip()}, 0]', code)
    code = re.sub(r'(\.shift\s*\()\[([^,\[\]]+),\s*([^,\[\]]+)\]',
                  lambda m: f'{m.group(1)}[{m.group(2).strip()}, {m.group(3).strip()}, 0]', code)

    # 8. Inject safe_tex wrapper and replace MathTex calls
    if 'def safe_tex' not in code:
        code = code.replace(
            'class MathVizScene(Scene):',
            MATHTEX_SAFE_WRAPPER + 'class MathVizScene(Scene):'
        )
    code = re.sub(r'\bMathTex\(', 'safe_tex(', code)

    # 9. Add numpy import
    if 'import numpy' not in code:
        code = code.replace('from manim import *', 'from manim import *\nimport numpy as np')

    # 10. Syntax check — fallback to safe scene if broken
    try:
        ast.parse(code)
    except SyntaxError as e:
        logger.error(f"Syntax error after all fixes, using fallback: {e}")
        code = '''from manim import *
import numpy as np

def safe_tex(latex_str, **kwargs):
    try:
        return MathTex(latex_str, **kwargs)
    except Exception:
        clean = latex_str.replace("\\\\", "").replace("{", "").replace("}", "")
        return Text(clean[:60], font_size=kwargs.get("font_size", 36))

class MathVizScene(Scene):
    def construct(self):
        title = Text("Math Visualization", font_size=48)
        subtitle = Text("Animation could not be generated", font_size=28, color=YELLOW)
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
    result = llm_call(
        system_prompt=SYSTEM_PROMPT + RESPONSE_FORMAT,
        user_prompt=user_prompt,
        response_model=AnimationCode,
        max_retries=2
    )

    result.python_code = _fix_common_issues(result.python_code)
    valid, err = _syntax_check(result.python_code)
    if not valid:
        raise ValueError(f"Generated Manim code has a syntax error: {err}")

    lines = len(result.python_code.strip().splitlines())
    logger.info("Animation code generated: class=%s, lines=%d", result.manim_class_name, lines)
    return result