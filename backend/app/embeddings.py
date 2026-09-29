from google import genai
from google.genai import types

from app.config import settings

_client = None

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = get_client()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return [e.values for e in result.embeddings]
