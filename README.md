# GnomeFinance AI Analyst

An LLM-powered market analyst that combines **retrieval-augmented generation**, **tool-calling agents**, and **structured output** to answer questions about stocks, SEC filings, and market data.

Ask it `"How has NVDA performed this month and is it overbought?"` and it will:
1. Decide to call `get_price` and `compute_indicator` on its own
2. Fetch live market data via yfinance
3. Synthesize a factual answer with the numbers it retrieved

Ask it `"What did NVDA's 10-K say about data center revenue?"` and it will:
1. Retrieve relevant chunks from an ingested SEC filing
2. Call Gemini with strict JSON schema enforcement
3. Return a structured analysis with citations

## Stack

- **FastAPI** — HTTP API
- **Google Gemini** (`google-genai`) — chat + embeddings
- **SQLite + JSON vectors** — local storage for RAG
- **yfinance** — live market data
- **Python 3.14** on Termux/proot (aarch64)
