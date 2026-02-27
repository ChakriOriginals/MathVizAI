from __future__ import annotations
import logging
import io
import pdfplumber
from backend.llm_client import llm_call
from backend.models import ParsedContent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert mathematical content analyst. Extract structured information from mathematical text.
Return a JSON object with EXACTLY these keys:
{
  "main_topic": "<string>",
  "definitions": ["<list of key definitions>"],
  "key_equations": ["<list of equations in LaTeX>"],
  "core_claims": ["<list of main theorems or claims>"],
  "example_instances": ["<list of concrete examples>"]
}
Rules: All LaTeX must be valid inline LaTeX. Maximum 6 items per list. Use [] if absent.
"""

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    pages = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
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
    truncated = raw_text[:6000]
    if len(raw_text) > 6000:
        logger.warning("Input truncated from %d to 6000 chars.", len(raw_text))
    user_prompt = f"Difficulty level: {difficulty_level}\n\nInput content:\n{truncated}"
    result = llm_call(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, response_model=ParsedContent)
    logger.info("Parser agent completed. Topic: %s", result.main_topic)
    return result