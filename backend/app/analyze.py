import logging

from app.llm import chat_json
from app.rag.retrieve import retrieve

log = logging.getLogger("gnomefinance.analyze")

SYSTEM_PROMPT = """You are a financial analyst. You are given a user question and a set of retrieved context chunks from SEC filings, news, or market documents.

Rules:
- Answer ONLY using the provided context. If the context does not contain the answer, say so.
- Cite sources using the chunk numbers [1], [2], etc.
- Be concise and factual. No speculation.
- Return JSON matching this schema exactly:
{
  "summary": "one-paragraph answer",
  "key_points": ["point 1", "point 2"],
  "risks": ["risk 1"],
  "citations": [{"chunk_index": 1, "snippet": "quoted text"}]
}
"""


def _format_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        ticker = h.get("ticker") or "?"
        source = h.get("source") or "?"
        lines.append(f"[{i}] ({ticker} | {source})\n{h['content']}")
    return "\n\n".join(lines)


def analyze(query: str, top_k: int = 5, tickers: list[str] | None = None) -> dict:
    hits = retrieve(query, top_k=top_k, tickers=tickers)
    if not hits:
        return {
            "summary": "No relevant documents found in the knowledge base.",
            "key_points": [],
            "risks": [],
            "citations": [],
        }

    context = _format_context(hits)
    user_prompt = f"Question: {query}\n\nContext:\n{context}\n\nReturn the JSON analysis."

    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        log.exception("analyze failed")
        return {
            "summary": f"Analysis failed: {e}",
            "key_points": [],
            "risks": [],
            "citations": [],
        }

    result.setdefault("summary", "")
    result.setdefault("key_points", [])
    result.setdefault("risks", [])
    result.setdefault("citations", [])
    return result
