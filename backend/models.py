from __future__ import annotations
import re
from typing import Any, List, Optional
from pydantic import BaseModel, Field

class ParsedContent(BaseModel):
    main_topic: str
    definitions: List[str] = Field(default_factory=list)
    key_equations: List[str] = Field(default_factory=list)
    core_claims: List[str] = Field(default_factory=list)
    example_instances: List[str] = Field(default_factory=list)

class Concept(BaseModel):
    concept_name: str
    intuitive_explanation: str
    mathematical_form: str
    why_it_matters: str

class ConceptExtractionResult(BaseModel):
    core_concepts: List[Concept]
    concept_ordering: List[str]

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

class ManimObject(BaseModel):
    obj_id: str
    obj_type: str
    properties: dict

class ManimAnimation(BaseModel):
    action: str
    target: str
    duration: float = 1.0
    kwargs: dict = Field(default_factory=dict)

class SceneInstruction(BaseModel):
    scene_id: int
    objects: List[ManimObject] = Field(default_factory=list)
    animations: List[ManimAnimation] = Field(default_factory=list)
    camera_actions: List[Any] = Field(default_factory=list)  # Accept anything, ignored at render

class SceneInstructionSet(BaseModel):
    scene_instructions: List[SceneInstruction]

class AnimationCode(BaseModel):
    python_code: str = ""
    manim_class_name: str = "MathVizScene"
    code: str = ""

    def model_post_init(self, __context):
        if self.code and not self.python_code:
            self.python_code = self.code
        if not self.manim_class_name:
            self.manim_class_name = "MathVizScene"
        if self.python_code:
            match = re.search(r'class\s+(\w+)\s*\(\s*Scene\s*\)', self.python_code)
            if match:
                self.manim_class_name = match.group(1)

class RenderResult(BaseModel):
    video_path: Optional[str] = None
    render_status: str
    error_log: str = ""

class GenerateVideoRequest(BaseModel):
    topic_or_text: str = Field(..., min_length=3, max_length=8000)
    difficulty_level: str = Field(default="undergraduate", pattern="^(high_school|undergraduate)$")

class GenerateVideoResponse(BaseModel):
    job_id: str
    status: str
    video_path: Optional[str] = None
    error: Optional[str] = None
    pipeline_trace: Optional[dict] = None