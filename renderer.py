"""
Renderer Module
Saves the generated Manim script to a temp file, runs the Manim CLI,
and returns the path to the rendered video.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from backend.config import settings
from backend.models import AnimationCode, RenderResult

logger = logging.getLogger(__name__)

# Manim quality flag mapping
QUALITY_FLAGS = {
    "low_quality": "-ql",
    "medium_quality": "-qm",
    "high_quality": "-qh",
    "production_quality": "-qp",
}


def run(animation: AnimationCode, job_id: str | None = None) -> RenderResult:
    """
    Render the Manim script to an MP4 video.

    Returns a RenderResult with video_path on success or error_log on failure.
    """
    job_id = job_id or str(uuid.uuid4())
    quality_flag = QUALITY_FLAGS.get(settings.render_quality, "-qm")

    # Write script to a temp file
    script_path = settings.temp_dir / f"scene_{job_id}.py"
    try:
        script_path.write_text(animation.python_code, encoding="utf-8")
    except IOError as exc:
        return RenderResult(
            render_status="failure",
            error_log=f"Failed to write Manim script: {exc}",
        )

    # Build Manim CLI command
    # manim render <quality> <file> <ClassName> --media_dir <output>
    media_dir = settings.output_dir / job_id
    media_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "manim",
        "render",
        quality_flag,
        str(script_path),
        animation.manim_class_name,
        "--media_dir", str(media_dir),
        "--disable_caching",
    ]

    logger.info("Running Manim: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.render_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return RenderResult(
            render_status="failure",
            error_log=(
                f"Manim render timed out after {settings.render_timeout_seconds}s. "
                "Try reducing scene complexity."
            ),
        )
    except FileNotFoundError:
        return RenderResult(
            render_status="failure",
            error_log=(
                "Manim executable not found. "
                "Install it with: pip install manim"
            ),
        )

    if proc.returncode != 0:
        error_log = f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        logger.error("Manim render failed (rc=%d):\n%s", proc.returncode, error_log)
        return RenderResult(render_status="failure", error_log=error_log)

    # Locate output video (Manim places it in media_dir/videos/.../quality/*.mp4)
    video_path = _find_output_video(media_dir)
    if video_path is None:
        return RenderResult(
            render_status="failure",
            error_log=(
                f"Manim reported success but no .mp4 found under {media_dir}.\n"
                f"STDOUT:\n{proc.stdout}"
            ),
        )

    # Copy to a predictable output location
    final_path = settings.output_dir / f"{job_id}.mp4"
    shutil.copy2(video_path, final_path)

    logger.info("Render successful: %s", final_path)
    return RenderResult(
        video_path=str(final_path),
        render_status="success",
    )


def _find_output_video(media_dir: Path) -> Path | None:
    """Recursively search for the most recently modified .mp4 under media_dir."""
    mp4_files = list(media_dir.rglob("*.mp4"))
    if not mp4_files:
        return None
    # Return the most recently modified file
    return max(mp4_files, key=lambda p: p.stat().st_mtime)
