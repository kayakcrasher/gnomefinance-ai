import json
import logging
import time

from google import genai
from google.genai import types

from app.config import settings

log = logging.getLogger("gnomefinance.llm")

_client = None
MODEL_CHAIN = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
]
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


def _try_model(model: str, system: str, user: str) -> str:
    client = _get_client()
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return resp.text


def chat_json(system: str, user: str) -> dict:
    last_err: Exception | None = None

    for model in MODEL_CHAIN:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                text = _try_model(model, system, user)
                return json.loads(text)
            except json.JSONDecodeError as e:
                log.error("Invalid JSON from %s: %s", model, e)
                last_err = e
                break
            except Exception as e:
                log.warning("model=%s attempt=%d failed: %s", model, attempt, e)
                last_err = e
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_SECONDS * attempt)
                else:
                    break

    raise RuntimeError(f"All models failed. Last error: {last_err}")
