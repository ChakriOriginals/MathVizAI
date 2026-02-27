from __future__ import annotations
import json
import logging
from typing import Type, TypeVar
import anthropic
from pydantic import BaseModel, ValidationError
from backend.config import settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
_client: anthropic.Anthropic | None = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client

STRICT_JSON_SUFFIX = "\n\nCRITICAL: Your response MUST be valid JSON only. No markdown fences, no prose, no explanation — raw JSON only."

def llm_call(system_prompt: str, user_prompt: str, response_model: Type[T], max_retries: int = 2) -> T:
    client = _get_client()
    last_error = None

    for attempt in range(max_retries + 1):
        suffix = STRICT_JSON_SUFFIX if attempt > 0 else ""
        try:
            logger.debug("LLM call attempt %d for %s", attempt + 1, response_model.__name__)
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                system=system_prompt + suffix + "\n\nYou must respond with valid JSON only. No markdown, no backticks.",
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )
            raw = response.content[0].text.strip()

            # Strip any accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                raw = raw.rsplit("```", 1)[0]

            data = json.loads(raw)
            return response_model.model_validate(data)

        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning("LLM parse error (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)

    raise RuntimeError(
        f"LLM failed to return valid {response_model.__name__} "
        f"after {max_retries + 1} attempts. Last error: {last_error}"
    )