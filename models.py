"""
Shared Pydantic models used throughout the MathVizAI pipeline.
Strict typing prevents silent data corruption between agents.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ─── Parser Agent Output ────────────────────────────────────────────────────

class ParsedContent(BaseModel):
    main_topic: str
    definitions: List[str] = Field(default_factory=list)
    key_equations: List[str] = Field(default_factory=list)  # LaTeX strings
    core_claims: List[str] = Field(default_factory=list)
    example_instances: List[str] = Field(default_factory=list)


# ─── Concept Extraction Output ───────────────────────────────────────────────

class Concept(BaseModel):
    concept_name: str
    intuitive_explanation: str
    mathematical_form: str  # LaTeX
    why_it_matters: str


class ConceptExtractionResult(BaseModel):
    core_concepts: List[Concept]
    concept_ordering: List[str]  # Ordered list of concept_name strings


# ─── Pedagogical Planner Output ──────────────────────────────────────────────

class Scene(BaseModel):
    scene_id: int
    scene_title: str
    learning_goal: str
    visual_metaphor: str
    equations_to_show: List[str] = Field(default_factory=list)
    animation_strategy: str
    estimated_duration_seconds: int = 40


class PedagogyPlan(BaseModel):
    scenes: List[Scene]


# ─── Scene Generator Output ──────────────────────────────────────────────────

class ManimObject(BaseModel):
    obj_id: str            # unique reference name used in animations
    obj_type: str          # Axes | NumberLine | Text | MathTex | Graph
    properties: dict       # free-form dict passed to Manim constructor


class ManimAnimation(BaseModel):
    action: str            # Create | Transform | FadeIn | FadeOut | GrowFromCenter | Write
    target: str            # obj_id reference
    duration: float = 1.0
    kwargs: dict = Field(default_factory=dict)


class SceneInstruction(BaseModel):
    scene_id: int
    objects: List[ManimObject] = Field(default_factory=list)
    animations: List[ManimAnimation] = Field(default_factory=list)
    camera_actions: List[str] = Field(default_factory=list)


class SceneInstructionSet(BaseModel):
    scene_instructions: List[SceneInstruction]


# ─── Animation Code Generator Output ────────────────────────────────────────

class AnimationCode(BaseModel):
    manim_class_name: str
    python_code: str


# ─── Renderer Output ─────────────────────────────────────────────────────────

class RenderResult(BaseModel):
    video_path: Optional[str] = None
    render_status: str   # "success" | "failure"
    error_log: str = ""


# ─── Top-level Job Models ─────────────────────────────────────────────────────

class GenerateVideoRequest(BaseModel):
    topic_or_text: str = Field(..., min_length=3, max_length=8000,
                               description="Math topic name or short excerpt (≤10 pages)")
    difficulty_level: str = Field(default="undergraduate",
                                  pattern="^(high_school|undergraduate)$")


class GenerateVideoResponse(BaseModel):
    job_id: str
    status: str
    video_path: Optional[str] = None
    error: Optional[str] = None
    pipeline_trace: Optional[dict] = None  # Debug: intermediate agent outputs
