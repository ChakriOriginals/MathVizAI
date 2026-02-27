"""
Pipeline Orchestrator
Runs all agents in sequence and returns the final result.
Each step is logged and errors produce clear, actionable messages.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from backend.agents import (
    animation_agent,
    concept_agent,
    parser_agent,
    pedagogy_agent,
    scene_agent,
)
from backend.models import (
    AnimationCode,
    ConceptExtractionResult,
    GenerateVideoResponse,
    ParsedContent,
    PedagogyPlan,
    RenderResult,
    SceneInstructionSet,
)
from backend.modules import math_validator, renderer

logger = logging.getLogger(__name__)


@dataclass
class PipelineTrace:
    """Stores intermediate outputs from each agent for debugging."""
    parsed_content: Optional[ParsedContent] = None
    concepts: Optional[ConceptExtractionResult] = None
    pedagogy_plan: Optional[PedagogyPlan] = None
    scene_instructions: Optional[SceneInstructionSet] = None
    animation_code: Optional[AnimationCode] = None
    render_result: Optional[RenderResult] = None
    errors: list[str] = field(default_factory=list)


def run_pipeline(
    raw_text: str,
    difficulty_level: str = "undergraduate",
    job_id: Optional[str] = None,
) -> GenerateVideoResponse:
    """
    Execute the full MathVizAI pipeline synchronously.

    Steps:
        1. Parse input
        2. Extract concepts
        3. Plan pedagogy
        4. Generate scene instructions
        5. Generate Manim code
        6. Render video

    Returns a GenerateVideoResponse with status and video_path.
    """
    job_id = job_id or str(uuid.uuid4())
    trace = PipelineTrace()

    logger.info("=== MathVizAI Pipeline START  job_id=%s ===", job_id)

    # ── Step 1: Parse ────────────────────────────────────────────────────────
    try:
        logger.info("[1/6] Running Parser Agent...")
        trace.parsed_content = parser_agent.run(raw_text, difficulty_level)
    except Exception as exc:
        msg = f"Parser Agent failed: {exc}"
        logger.error(msg)
        return GenerateVideoResponse(job_id=job_id, status="failed", error=msg)

    # ── Step 2: Concept Extraction ───────────────────────────────────────────
    try:
        logger.info("[2/6] Running Concept Extraction Agent...")
        trace.concepts = concept_agent.run(trace.parsed_content, difficulty_level)
    except Exception as exc:
        msg = f"Concept Extraction Agent failed: {exc}"
        logger.error(msg)
        return GenerateVideoResponse(job_id=job_id, status="failed", error=msg)

    # ── Step 3: Pedagogy Planning ─────────────────────────────────────────────
    try:
        logger.info("[3/6] Running Pedagogy Planner Agent...")
        trace.pedagogy_plan = pedagogy_agent.run(trace.concepts, difficulty_level)
    except Exception as exc:
        msg = f"Pedagogy Planner Agent failed: {exc}"
        logger.error(msg)
        return GenerateVideoResponse(job_id=job_id, status="failed", error=msg)

    # ── Step 3b: Validate equations in plan ──────────────────────────────────
    for scene in trace.pedagogy_plan.scenes:
        if scene.equations_to_show:
            scene.equations_to_show = math_validator.filter_valid_equations(
                scene.equations_to_show
            )

    # ── Step 4: Scene Generation ──────────────────────────────────────────────
    try:
        logger.info("[4/6] Running Scene Generator Agent...")
        trace.scene_instructions = scene_agent.run(trace.pedagogy_plan)
    except Exception as exc:
        msg = f"Scene Generator Agent failed: {exc}"
        logger.error(msg)
        return GenerateVideoResponse(job_id=job_id, status="failed", error=msg)

    # ── Step 5: Animation Code Generation ────────────────────────────────────
    try:
        logger.info("[5/6] Running Animation Code Generator Agent...")
        trace.animation_code = animation_agent.run(
            trace.scene_instructions, trace.pedagogy_plan
        )
    except ValueError as exc:
        # Syntax error in generated code — surface to user
        msg = f"Animation Code Generator produced invalid code: {exc}"
        logger.error(msg)
        return GenerateVideoResponse(job_id=job_id, status="failed", error=msg)
    except Exception as exc:
        msg = f"Animation Code Generator Agent failed: {exc}"
        logger.error(msg)
        return GenerateVideoResponse(job_id=job_id, status="failed", error=msg)

    # ── Step 6: Render ────────────────────────────────────────────────────────
    try:
        logger.info("[6/6] Running Renderer...")
        trace.render_result = renderer.run(trace.animation_code, job_id=job_id)
    except Exception as exc:
        msg = f"Renderer crashed unexpectedly: {exc}"
        logger.error(msg)
        return GenerateVideoResponse(job_id=job_id, status="failed", error=msg)

    if trace.render_result.render_status == "failure":
        return GenerateVideoResponse(
            job_id=job_id,
            status="failed",
            error=f"Manim render failed:\n{trace.render_result.error_log}",
            pipeline_trace=_trace_to_dict(trace),
        )

    logger.info("=== MathVizAI Pipeline COMPLETE  job_id=%s ===", job_id)
    return GenerateVideoResponse(
        job_id=job_id,
        status="success",
        video_path=trace.render_result.video_path,
        pipeline_trace=_trace_to_dict(trace),
    )


def _trace_to_dict(trace: PipelineTrace) -> dict:
    """Serialize pipeline trace to a JSON-safe dict for debugging."""
    return {
        "parsed_content": trace.parsed_content.model_dump() if trace.parsed_content else None,
        "concepts": trace.concepts.model_dump() if trace.concepts else None,
        "pedagogy_plan": trace.pedagogy_plan.model_dump() if trace.pedagogy_plan else None,
        "scene_instructions": trace.scene_instructions.model_dump() if trace.scene_instructions else None,
        "animation_code": {
            "class_name": trace.animation_code.manim_class_name,
            "lines": len(trace.animation_code.python_code.splitlines()),
        } if trace.animation_code else None,
        "render_result": trace.render_result.model_dump() if trace.render_result else None,
        "errors": trace.errors,
    }
