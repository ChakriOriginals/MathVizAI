"""
Central configuration for MathVizAI backend.
All settings are driven by environment variables with safe defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    llm_max_tokens: int = 4000
    llm_temperature: float = 0.3  # Low temperature for structured/deterministic outputs

    # Pipeline limits
    max_scenes: int = 5
    max_concepts: int = 5
    max_input_pages: int = 10
    max_manim_lines: int = 400

    # Rendering
    render_quality: str = "medium_quality"   # low_quality | medium_quality | high_quality
    render_resolution: str = "720p"
    render_fps: int = 30
    render_timeout_seconds: int = 300        # 5 min hard limit per render

    # Storage
    output_dir: Path = Path("./outputs")
    temp_dir: Path = Path("./tmp")

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
