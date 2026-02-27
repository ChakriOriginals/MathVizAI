"""
LLM client wrapper.
- Enforces JSON-mode responses
- Retries with stricter prompt on parse failure
- Validates against a Pydantic model
"""

from __future__ import annotations

import json
import logging
from typing import Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from backend.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Please add it to your .env file or environment."
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


STRICT_JSON_SUFFIX = (
    "\n\nCRITICAL: Your response MUST be valid JSON only. "
    "No markdown fences, no prose, no explanation — raw JSON only."
)


def llm_call(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    max_retries: int = 2,
) -> T:
    """
    Call the LLM and parse the response into `response_model`.
    On JSON parse or validation failure, retry up to `max_retries` times
    with progressively stricter prompting.
    """
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        suffix = STRICT_JSON_SUFFIX if attempt > 0 else ""
        try:
            logger.debug("LLM call attempt %d for %s", attempt + 1, response_model.__name__)
            response = client.chat.completions.create(
                model=settings.openai_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt + suffix},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            return response_model.model_validate(data)

        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "LLM response parse error (attempt %d/%d): %s",
                attempt + 1, max_retries + 1, exc,
            )

    raise RuntimeError(
        f"LLM failed to return valid {response_model.__name__} "
        f"after {max_retries + 1} attempts. Last error: {last_error}"
    )
