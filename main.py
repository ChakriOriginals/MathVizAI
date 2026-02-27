"""
MathVizAI FastAPI Application
Exposes REST endpoints to trigger the pipeline and retrieve results.
"""

from __future__ import annotations

import logging
import mimetypes
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.agents.parser_agent import extract_text_from_pdf
from backend.config import settings
from backend.models import GenerateVideoRequest, GenerateVideoResponse
from backend.pipeline import run_pipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MathVizAI API",
    description="Generate Manim math animations from topics or PDF excerpts.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "version": "0.1.0"}


# ── Main Endpoint: Generate from text/topic ───────────────────────────────────

@app.post("/generate-video", response_model=GenerateVideoResponse, tags=["pipeline"])
def generate_video(request: GenerateVideoRequest):
    """
    Run the full MathVizAI pipeline on a text topic or short excerpt.

    - `topic_or_text`: plain text topic (e.g. "Central Limit Theorem") or a paragraph excerpt
    - `difficulty_level`: "high_school" or "undergraduate"
    """
    job_id = str(uuid.uuid4())
    logger.info("New job %s | difficulty=%s | input_len=%d",
                job_id, request.difficulty_level, len(request.topic_or_text))

    result = run_pipeline(
        raw_text=request.topic_or_text,
        difficulty_level=request.difficulty_level,
        job_id=job_id,
    )
    return result


# ── PDF Upload Endpoint ───────────────────────────────────────────────────────

@app.post("/generate-video-from-pdf", response_model=GenerateVideoResponse, tags=["pipeline"])
async def generate_video_from_pdf(
    file: UploadFile = File(..., description="PDF file (max 10 pages)"),
    difficulty_level: str = Form(default="undergraduate"),
):
    """
    Upload a PDF (≤10 pages) and run the full pipeline.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 20 * 1024 * 1024:  # 20 MB guard
        raise HTTPException(status_code=400, detail="PDF file exceeds 20 MB limit.")

    try:
        raw_text = extract_text_from_pdf(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    job_id = str(uuid.uuid4())
    logger.info("New PDF job %s | difficulty=%s | text_len=%d",
                job_id, difficulty_level, len(raw_text))

    result = run_pipeline(
        raw_text=raw_text,
        difficulty_level=difficulty_level,
        job_id=job_id,
    )
    return result


# ── Download Endpoint ─────────────────────────────────────────────────────────

@app.get("/download/{job_id}", tags=["pipeline"])
def download_video(job_id: str):
    """Download the rendered .mp4 for a completed job."""
    # Basic input sanitization to prevent path traversal
    if not job_id.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid job_id format.")

    video_path = settings.output_dir / f"{job_id}.mp4"
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video not found for job {job_id}. "
                   "The job may have failed or not yet completed.",
        )

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"mathvizai_{job_id}.mp4",
    )
