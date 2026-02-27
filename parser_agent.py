"""
Agent 1: Paper / Topic Parser
Converts raw user input (topic string or PDF text) into structured mathematical content.
"""

from __future__ import annotations

import logging

import pdfplumber

from backend.llm_client import llm_call
from backend.models import ParsedContent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert mathematical content analyst. Your job is to extract structured
information from mathematical text.

Return a JSON object with EXACTLY these keys:
{
  "main_topic": "<string: the central topic>",
  "definitions": ["<list of key definitions as plain English strings>"],
  "key_equations": ["<list of important equations in LaTeX notation>"],
  "core_claims": ["<list of main theorems or claims as plain English>"],
  "example_instances": ["<list of concrete examples or applications>"]
}

Rules:
- All LaTeX must be valid inline LaTeX (wrap in $...$)
- Definitions and claims must be plain, jargon-light English
- Maximum 6 items per list
- If information is absent, use an empty list []
"""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF byte stream using pdfplumber."""
    pages: list[str] = []
    try:
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Honour MVP constraint: max 10 pages
            for page in pdf.pages[:10]:
                text = page.extract_text() or ""
                pages.append(text)
    except Exception as exc:
        raise ValueError(f"Failed to extract text from PDF: {exc}") from exc

    combined = "\n\n".join(pages).strip()
    if not combined:
        raise ValueError("PDF appears to be empty or contains only scanned images.")
    return combined


def run(raw_text: str, difficulty_level: str = "undergraduate") -> ParsedContent:
    """
    Parse raw text (already extracted from PDF or typed by user) into
    a structured ParsedContent object.
    """
    # Truncate to ~6000 chars to stay within token budget
    truncated = raw_text[:6000]
    if len(raw_text) > 6000:
        logger.warning("Input truncated from %d to 6000 chars.", len(raw_text))

    user_prompt = (
        f"Difficulty level: {difficulty_level}\n\n"
        f"Input content:\n{truncated}"
    )

    result = llm_call(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=ParsedContent,
    )
    logger.info("Parser agent completed. Topic: %s", result.main_topic)
    return result
