import json

import numpy as np

from app.db import get_conn
from app.embeddings import embed


def _cosine(query_vec, matrix):
    q = np.array(query_vec, dtype=np.float32)
    m = np.array(matrix, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-12)
    mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    return mn @ qn


def retrieve(query: str, top_k: int = 5, tickers: list[str] | None = None) -> list[dict]:
    with get_conn() as conn:
        sql = """
            SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding,
                   d.source, d.ticker
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
        """
        params: list = []
        if tickers:
            placeholders = ",".join("?" * len(tickers))
            sql += f" WHERE d.ticker IN ({placeholders})"
            params = [t.upper() for t in tickers]
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return []

    query_vec = embed([query])[0]
    matrix = [json.loads(r["embedding"]) for r in rows]
    scores = _cosine(query_vec, matrix)
    order = np.argsort(-scores)[:top_k]

    results = []
    for idx in order:
        r = rows[int(idx)]
        results.append({
            "score": float(scores[idx]),
            "content": r["content"],
            "source": r["source"],
            "ticker": r["ticker"],
            "chunk_index": r["chunk_index"],
        })
    return results
