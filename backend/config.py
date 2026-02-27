from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_max_tokens: int = 8000
    llm_temperature: float = 0.3
    max_scenes: int = 5
    max_concepts: int = 5
    max_input_pages: int = 10
    max_manim_lines: int = 400
    render_quality: str = "medium_quality"
    render_resolution: str = "720p"
    render_fps: int = 30
    render_timeout_seconds: int = 300
    output_dir: Path = Path("D:/1IU/Practice/LLM Project Practice/outputs")
    temp_dir: Path = Path("D:/1IU/Practice/LLM Project Practice/tmp")
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()