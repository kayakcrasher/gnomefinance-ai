import json
import re

import httpx

from app.db import get_conn
from app.embeddings import embed

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
USER_AGENT = "GnomeFinance/0.1 (contact@example.com)"

EDGAR_CIKS = {
    "NVDA": 1045810,
    "AMD": 2488,
    "AAPL": 320193,
    "MSFT": 789019,
    "GOOGL": 1652044,
    "META": 1326801,
}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def strip_html(html: str) -> str:
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def strip_html(html: str) -> str:
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html).strip()


def ingest_text(source: str, content: str, ticker: str | None = None, metadata: dict | None = None) -> int:
    chunks = chunk_text(content)
    if not chunks:
        return 0
    vectors = embed(chunks)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (source, ticker, content, metadata) VALUES (?, ?, ?, ?)",
            (source, ticker, content, json.dumps(metadata or {})),
        )
        doc_id = cur.lastrowid
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            conn.execute(
                "INSERT INTO chunks (document_id, chunk_index, content, embedding, metadata) VALUES (?, ?, ?, ?, ?)",
                (doc_id, i, chunk, json.dumps(vec), json.dumps(metadata or {})),
            )
        conn.commit()
    return len(chunks)


def ingest_url(url: str, ticker: str | None = None) -> int:
    r = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    text = strip_html(r.text) if "<html" in r.text[:2000].lower() else r.text
    return ingest_text(source=url, content=text, ticker=ticker, metadata={"url": url})


def ingest_edgar(ticker: str, form: str = "10-K", limit: int = 1) -> int:
    cik = EDGAR_CIKS.get(ticker.upper())
    if not cik:
        raise ValueError(f"Unknown ticker: {ticker}")
    headers = {"User-Agent": USER_AGENT}
    sub = httpx.get(
        f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
        headers=headers,
        timeout=30.0,
    ).json()
    recent = sub["filings"]["recent"]
    total = 0
    matched = 0
    for i, f in enumerate(recent["form"]):
        if f != form:
            continue
        accession = recent["accessionNumber"][i].replace("-", "")
        primary = recent["primaryDocument"][i]
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary}"
        try:
            total += ingest_url(doc_url, ticker=ticker.upper())
            matched += 1
        except Exception as e:
            print(f"skip {doc_url}: {e}")
        if matched >= limit:
            break
    return total
