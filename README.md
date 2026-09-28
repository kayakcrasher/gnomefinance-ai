
# GnomeFinance AI Analyst

LLM-powered market analysis with RAG, tool calling, and evaluation.

## Demo
[GIF of asking "Compare NVDA and AMD momentum" → structured analysis]

## Architecture
[Diagram]

## Key Features
- RAG over SEC filings, news, earnings transcripts
- Tool calling: live prices, technical indicators, financials
- Structured outputs with citations
- Evaluation harness (retrieval, faithfulness, tool accuracy)
- Streaming responses, cost tracking, guardrails

## Eval Results
| Metric | Score |
|--------|-------|
| Retrieval Hit Rate | 0.87 |
| Faithfulness | 0.92 |
| Tool Accuracy | 0.89 |
| Avg Latency | 3.2s |
| Avg Cost/Query | $0.004 |

## Quick Start
docker compose up

## What I Learned
- Chunking strategy matters more than model choice for RAG
- Tool calling accuracy improves with few-shot examples in system prompt
- Evaluation is the hardest part — built custom metrics for financial domain
