"""
MathVizAI Test Suite
Tests all agents, validators, and the pipeline orchestrator.
Uses mocking so tests run without real API keys or Manim installed.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import (
    AnimationCode,
    Concept,
    ConceptExtractionResult,
    GenerateVideoRequest,
    ManimAnimation,
    ManimObject,
    ParsedContent,
    PedagogyPlan,
    RenderResult,
    Scene,
    SceneInstruction,
    SceneInstructionSet,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_parsed_content() -> ParsedContent:
    return ParsedContent(
        main_topic="Central Limit Theorem",
        definitions=["A random variable is a function from a sample space to real numbers."],
        key_equations=["$\\bar{X}_n \\xrightarrow{d} N(\\mu, \\sigma^2/n)$"],
        core_claims=["The sum of i.i.d. random variables converges in distribution to a normal."],
        example_instances=["Coin flip sums, dice roll averages"],
    )


def make_concepts() -> ConceptExtractionResult:
    return ConceptExtractionResult(
        core_concepts=[
            Concept(
                concept_name="Sample Mean",
                intuitive_explanation="Average of n observations",
                mathematical_form="$\\bar{X} = \\frac{1}{n}\\sum_{i=1}^n X_i$",
                why_it_matters="Foundation of estimation",
            ),
            Concept(
                concept_name="Convergence in Distribution",
                intuitive_explanation="Histogram shape approaches bell curve",
                mathematical_form="$\\bar{X}_n \\xrightarrow{d} N(\\mu, \\sigma^2/n)$",
                why_it_matters="Justifies using normal approximations",
            ),
        ],
        concept_ordering=["Sample Mean", "Convergence in Distribution"],
    )


def make_pedagogy_plan() -> PedagogyPlan:
    return PedagogyPlan(
        scenes=[
            Scene(
                scene_id=1,
                scene_title="The Averaging Intuition",
                learning_goal="Understand why averaging reduces randomness",
                visual_metaphor="Coins being flipped, histogram building up",
                equations_to_show=[],
                animation_strategy="Show dots appearing on a number line, histogram grows",
                estimated_duration_seconds=40,
            ),
            Scene(
                scene_id=2,
                scene_title="The Bell Curve Emerges",
                learning_goal="See CLT visually",
                visual_metaphor="Normal curve morphing from uniform distribution",
                equations_to_show=["$\\bar{X}_n \\to N(\\mu, \\sigma^2/n)$"],
                animation_strategy="Histogram with many bars morphs into smooth bell curve",
                estimated_duration_seconds=50,
            ),
        ]
    )


def make_scene_instructions() -> SceneInstructionSet:
    return SceneInstructionSet(
        scene_instructions=[
            SceneInstruction(
                scene_id=1,
                objects=[
                    ManimObject(obj_id="title_1", obj_type="Text", properties={"text": "Central Limit Theorem"}),
                    ManimObject(obj_id="axes_1", obj_type="Axes", properties={"x_range": [-3, 3, 1], "y_range": [0, 1, 0.5]}),
                ],
                animations=[
                    ManimAnimation(action="Write", target="title_1", duration=1.0),
                    ManimAnimation(action="Create", target="axes_1", duration=1.5),
                ],
                camera_actions=[],
            )
        ]
    )


def make_animation_code() -> AnimationCode:
    return AnimationCode(
        manim_class_name="MathVizScene",
        python_code=(
            "from manim import *\n\n"
            "class MathVizScene(Scene):\n"
            "    def construct(self):\n"
            "        title = Text('Central Limit Theorem')\n"
            "        self.play(Write(title))\n"
            "        self.wait(2)\n"
        ),
    )


# ── Model Validation Tests ─────────────────────────────────────────────────────

class TestModels(unittest.TestCase):

    def test_generate_video_request_valid(self):
        req = GenerateVideoRequest(topic_or_text="Fourier Transform", difficulty_level="undergraduate")
        self.assertEqual(req.difficulty_level, "undergraduate")

    def test_generate_video_request_invalid_difficulty(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GenerateVideoRequest(topic_or_text="foo", difficulty_level="expert")

    def test_generate_video_request_too_short(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GenerateVideoRequest(topic_or_text="ab", difficulty_level="undergraduate")

    def test_parsed_content_defaults(self):
        p = ParsedContent(main_topic="Test Topic")
        self.assertEqual(p.definitions, [])
        self.assertEqual(p.key_equations, [])

    def test_pedagogy_plan_scene_ids(self):
        plan = make_pedagogy_plan()
        self.assertEqual(plan.scenes[0].scene_id, 1)
        self.assertEqual(plan.scenes[1].scene_id, 2)


# ── Math Validator Tests ──────────────────────────────────────────────────────

class TestMathValidator(unittest.TestCase):

    def test_filter_valid_equations(self):
        from backend.modules.math_validator import filter_valid_equations
        # Always returns input (may skip invalid ones) — test it returns a list
        result = filter_valid_equations(["$x^2 + y^2 = r^2$"])
        self.assertIsInstance(result, list)

    def test_validate_empty_string(self):
        from backend.modules.math_validator import validate_equation
        valid, err = validate_equation("")
        self.assertFalse(valid)
        self.assertIn("Empty", err)

    def test_strip_delimiters(self):
        from backend.modules.math_validator import _strip_delimiters
        self.assertEqual(_strip_delimiters("$x=1$"), "x=1")
        self.assertEqual(_strip_delimiters("$$x=1$$"), "x=1")


# ── Parser Agent Tests ────────────────────────────────────────────────────────

class TestParserAgent(unittest.TestCase):

    @patch("backend.agents.parser_agent.llm_call")
    def test_run_returns_parsed_content(self, mock_llm):
        from backend.agents import parser_agent
        mock_llm.return_value = make_parsed_content()
        result = parser_agent.run("Central Limit Theorem")
        self.assertIsInstance(result, ParsedContent)
        self.assertEqual(result.main_topic, "Central Limit Theorem")
        mock_llm.assert_called_once()

    @patch("backend.agents.parser_agent.llm_call")
    def test_input_truncated_at_6000_chars(self, mock_llm):
        from backend.agents import parser_agent
        mock_llm.return_value = make_parsed_content()
        long_text = "x" * 10000
        parser_agent.run(long_text)
        call_args = mock_llm.call_args
        user_prompt = call_args[1]["user_prompt"] if "user_prompt" in call_args[1] else call_args[0][1]
        # The truncated text should be inside the prompt
        self.assertLessEqual(len(user_prompt), 7000)  # some slack for prefix

    def test_pdf_extraction_empty_raises(self):
        from backend.agents.parser_agent import extract_text_from_pdf
        with self.assertRaises(ValueError):
            extract_text_from_pdf(b"")


# ── Concept Agent Tests ───────────────────────────────────────────────────────

class TestConceptAgent(unittest.TestCase):

    @patch("backend.agents.concept_agent.llm_call")
    def test_run_returns_concepts(self, mock_llm):
        from backend.agents import concept_agent
        mock_llm.return_value = make_concepts()
        result = concept_agent.run(make_parsed_content())
        self.assertIsInstance(result, ConceptExtractionResult)
        self.assertLessEqual(len(result.core_concepts), 5)

    @patch("backend.agents.concept_agent.llm_call")
    def test_max_concepts_enforced(self, mock_llm):
        from backend.agents import concept_agent
        from backend.config import settings
        # Return 7 concepts — should be trimmed to settings.max_concepts
        many = ConceptExtractionResult(
            core_concepts=[
                Concept(concept_name=f"C{i}", intuitive_explanation="x",
                        mathematical_form="$x$", why_it_matters="y")
                for i in range(7)
            ],
            concept_ordering=[f"C{i}" for i in range(7)],
        )
        mock_llm.return_value = many
        result = concept_agent.run(make_parsed_content())
        self.assertLessEqual(len(result.core_concepts), settings.max_concepts)


# ── Pedagogy Agent Tests ──────────────────────────────────────────────────────

class TestPedagogyAgent(unittest.TestCase):

    @patch("backend.agents.pedagogy_agent.llm_call")
    def test_run_returns_plan(self, mock_llm):
        from backend.agents import pedagogy_agent
        mock_llm.return_value = make_pedagogy_plan()
        result = pedagogy_agent.run(make_concepts())
        self.assertIsInstance(result, PedagogyPlan)
        self.assertGreaterEqual(len(result.scenes), 1)

    @patch("backend.agents.pedagogy_agent.llm_call")
    def test_scene_ids_renumbered(self, mock_llm):
        from backend.agents import pedagogy_agent
        plan = make_pedagogy_plan()
        plan.scenes[0].scene_id = 99  # Intentionally wrong
        mock_llm.return_value = plan
        result = pedagogy_agent.run(make_concepts())
        for i, s in enumerate(result.scenes):
            self.assertEqual(s.scene_id, i + 1)


# ── Scene Agent Tests ─────────────────────────────────────────────────────────

class TestSceneAgent(unittest.TestCase):

    @patch("backend.agents.scene_agent.llm_call")
    def test_run_returns_instructions(self, mock_llm):
        from backend.agents import scene_agent
        mock_llm.return_value = make_scene_instructions()
        result = scene_agent.run(make_pedagogy_plan())
        self.assertIsInstance(result, SceneInstructionSet)

    def test_sanitize_unknown_obj_type(self):
        from backend.agents.scene_agent import _sanitize_instruction
        instr = SceneInstruction(
            scene_id=1,
            objects=[ManimObject(obj_id="foo", obj_type="UnknownWidget", properties={})],
            animations=[ManimAnimation(action="FadeIn", target="foo", duration=1.0)],
        )
        sanitized = _sanitize_instruction(instr)
        self.assertEqual(sanitized.objects[0].obj_type, "Text")

    def test_sanitize_unknown_action(self):
        from backend.agents.scene_agent import _sanitize_instruction
        instr = SceneInstruction(
            scene_id=1,
            objects=[ManimObject(obj_id="bar", obj_type="Text", properties={"text": "hi"})],
            animations=[ManimAnimation(action="Teleport", target="bar", duration=1.0)],
        )
        sanitized = _sanitize_instruction(instr)
        self.assertEqual(sanitized.animations[0].action, "FadeIn")

    def test_sanitize_removes_orphan_animation(self):
        """Animation targeting a nonexistent obj_id should be dropped."""
        from backend.agents.scene_agent import _sanitize_instruction
        instr = SceneInstruction(
            scene_id=1,
            objects=[ManimObject(obj_id="real", obj_type="Text", properties={"text": "x"})],
            animations=[
                ManimAnimation(action="Write", target="real", duration=1.0),
                ManimAnimation(action="FadeIn", target="ghost_id", duration=1.0),
            ],
        )
        sanitized = _sanitize_instruction(instr)
        targets = [a.target for a in sanitized.animations]
        self.assertNotIn("ghost_id", targets)


# ── Animation Code Agent Tests ────────────────────────────────────────────────

class TestAnimationAgent(unittest.TestCase):

    @patch("backend.agents.animation_agent.llm_call")
    def test_run_returns_code(self, mock_llm):
        from backend.agents import animation_agent
        mock_llm.return_value = make_animation_code()
        result = animation_agent.run(make_scene_instructions(), make_pedagogy_plan())
        self.assertIn("from manim import", result.python_code)
        self.assertIn("class MathVizScene", result.python_code)

    @patch("backend.agents.animation_agent.llm_call")
    def test_invalid_syntax_raises(self, mock_llm):
        from backend.agents import animation_agent
        bad_code = AnimationCode(
            manim_class_name="MathVizScene",
            python_code="from manim import *\nclass Bad(Scene:\n    pass",
        )
        mock_llm.return_value = bad_code
        with self.assertRaises(ValueError):
            animation_agent.run(make_scene_instructions(), make_pedagogy_plan())

    def test_fix_show_creation(self):
        from backend.agents.animation_agent import _fix_common_issues
        code = "self.play(ShowCreation(circle))"
        fixed = _fix_common_issues(code)
        self.assertIn("Create(circle)", fixed)
        self.assertNotIn("ShowCreation", fixed)

    def test_fix_adds_import(self):
        from backend.agents.animation_agent import _fix_common_issues
        code = "class Foo(Scene):\n    pass"
        fixed = _fix_common_issues(code)
        self.assertIn("from manim import *", fixed)

    def test_syntax_check_valid(self):
        from backend.agents.animation_agent import _syntax_check
        ok, err = _syntax_check("x = 1 + 2")
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_syntax_check_invalid(self):
        from backend.agents.animation_agent import _syntax_check
        ok, err = _syntax_check("def foo(:\n    pass")
        self.assertFalse(ok)


# ── Renderer Tests ────────────────────────────────────────────────────────────

class TestRenderer(unittest.TestCase):

    @patch("backend.modules.renderer.subprocess.run")
    @patch("backend.modules.renderer._find_output_video")
    @patch("backend.modules.renderer.shutil.copy2")
    def test_successful_render(self, mock_copy, mock_find, mock_proc):
        from backend.modules import renderer
        mock_proc.return_value = MagicMock(returncode=0, stdout="", stderr="")
        fake_path = Path("/tmp/fake.mp4")
        mock_find.return_value = fake_path
        mock_copy.return_value = None

        result = renderer.run(make_animation_code(), job_id="test-job-123")
        self.assertEqual(result.render_status, "success")

    @patch("backend.modules.renderer.subprocess.run")
    def test_failed_render(self, mock_proc):
        from backend.modules import renderer
        mock_proc.return_value = MagicMock(
            returncode=1, stdout="", stderr="NameError: foo not defined"
        )

        result = renderer.run(make_animation_code(), job_id="test-fail")
        self.assertEqual(result.render_status, "failure")
        self.assertIn("NameError", result.error_log)

    @patch("backend.modules.renderer.subprocess.run")
    def test_manim_not_found(self, mock_proc):
        from backend.modules import renderer
        import subprocess
        mock_proc.side_effect = FileNotFoundError()

        result = renderer.run(make_animation_code(), job_id="test-notfound")
        self.assertEqual(result.render_status, "failure")
        self.assertIn("not found", result.error_log)

    @patch("backend.modules.renderer.subprocess.run")
    def test_render_timeout(self, mock_proc):
        from backend.modules import renderer
        import subprocess
        mock_proc.side_effect = subprocess.TimeoutExpired(cmd="manim", timeout=300)

        result = renderer.run(make_animation_code(), job_id="test-timeout")
        self.assertEqual(result.render_status, "failure")
        self.assertIn("timed out", result.error_log)


# ── Pipeline Integration Tests ─────────────────────────────────────────────────

class TestPipeline(unittest.TestCase):

    @patch("backend.pipeline.renderer.run")
    @patch("backend.pipeline.animation_agent.run")
    @patch("backend.pipeline.scene_agent.run")
    @patch("backend.pipeline.pedagogy_agent.run")
    @patch("backend.pipeline.concept_agent.run")
    @patch("backend.pipeline.parser_agent.run")
    def test_full_pipeline_success(
        self, mock_parser, mock_concept, mock_pedagogy,
        mock_scene, mock_animation, mock_renderer
    ):
        from backend.pipeline import run_pipeline

        mock_parser.return_value = make_parsed_content()
        mock_concept.return_value = make_concepts()
        mock_pedagogy.return_value = make_pedagogy_plan()
        mock_scene.return_value = make_scene_instructions()
        mock_animation.return_value = make_animation_code()
        mock_renderer.return_value = RenderResult(
            video_path="/outputs/test.mp4", render_status="success"
        )

        response = run_pipeline("Central Limit Theorem")
        self.assertEqual(response.status, "success")
        self.assertEqual(response.video_path, "/outputs/test.mp4")

    @patch("backend.pipeline.parser_agent.run")
    def test_pipeline_fails_on_parser_error(self, mock_parser):
        from backend.pipeline import run_pipeline
        mock_parser.side_effect = RuntimeError("LLM timeout")

        response = run_pipeline("Some topic")
        self.assertEqual(response.status, "failed")
        self.assertIn("Parser", response.error)

    @patch("backend.pipeline.renderer.run")
    @patch("backend.pipeline.animation_agent.run")
    @patch("backend.pipeline.scene_agent.run")
    @patch("backend.pipeline.pedagogy_agent.run")
    @patch("backend.pipeline.concept_agent.run")
    @patch("backend.pipeline.parser_agent.run")
    def test_pipeline_fails_on_render_failure(
        self, mock_parser, mock_concept, mock_pedagogy,
        mock_scene, mock_animation, mock_renderer
    ):
        from backend.pipeline import run_pipeline

        mock_parser.return_value = make_parsed_content()
        mock_concept.return_value = make_concepts()
        mock_pedagogy.return_value = make_pedagogy_plan()
        mock_scene.return_value = make_scene_instructions()
        mock_animation.return_value = make_animation_code()
        mock_renderer.return_value = RenderResult(
            render_status="failure", error_log="NameError: axes_1 undefined"
        )

        response = run_pipeline("Fourier Transform")
        self.assertEqual(response.status, "failed")
        self.assertIn("NameError", response.error)


# ── LLM Client Tests ──────────────────────────────────────────────────────────

class TestLLMClient(unittest.TestCase):

    @patch("backend.llm_client._get_client")
    def test_retries_on_json_error(self, mock_get_client):
        from backend.llm_client import llm_call

        bad_response = MagicMock()
        bad_response.choices = [MagicMock(message=MagicMock(content="not json {{"))]
        good_response = MagicMock()
        good_response.choices = [MagicMock(message=MagicMock(
            content='{"main_topic":"Test","definitions":[],"key_equations":[],"core_claims":[],"example_instances":[]}'
        ))]

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [bad_response, good_response]
        mock_get_client.return_value = mock_client

        result = llm_call("sys", "user", ParsedContent, max_retries=1)
        self.assertEqual(result.main_topic, "Test")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    @patch("backend.llm_client._get_client")
    def test_raises_after_max_retries(self, mock_get_client):
        from backend.llm_client import llm_call

        bad_response = MagicMock()
        bad_response.choices = [MagicMock(message=MagicMock(content="{invalid"))]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = bad_response
        mock_get_client.return_value = mock_client

        with self.assertRaises(RuntimeError):
            llm_call("sys", "user", ParsedContent, max_retries=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
