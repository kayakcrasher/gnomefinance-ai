import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import get_conn, init_db
from app.agents.graph import run_agent
from app.analyze import analyze
from app.rag.ingest import ingest_edgar, ingest_text, ingest_url
from app.rag.retrieve import retrieve
from app.schemas import (
    HealthResponse,
    IngestEdgarRequest,
    IngestResponse,
    IngestTextRequest,
    IngestUrlRequest,
    RetrieveHit,
    RetrieveRequest,
    RetrieveResponse,
    AnalysisRequest,
    AnalysisResponse,
    Citation,
    AgentRequest,
    AgentResponse,
    AgentStep,
)

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("gnomefinance")

app = FastAPI(title="GnomeFinance AI Analyst", version="0.2.0")


@app.on_event("startup")
def startup() -> None:
    log.info("Initializing database...")
    init_db()
    log.info("Database ready.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"
    return HealthResponse(status="ok", db=db_status)


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text_endpoint(req: IngestTextRequest) -> IngestResponse:
    n = ingest_text(source=req.source, content=req.content, ticker=req.ticker)
    return IngestResponse(chunks_created=n)


@app.post("/ingest/url", response_model=IngestResponse)
def ingest_url_endpoint(req: IngestUrlRequest) -> IngestResponse:
    try:
        n = ingest_url(req.url, ticker=req.ticker)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IngestResponse(chunks_created=n)


@app.post("/ingest/edgar", response_model=IngestResponse)
def ingest_edgar_endpoint(req: IngestEdgarRequest) -> IngestResponse:
    try:
        n = ingest_edgar(req.ticker, form=req.form, limit=req.limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IngestResponse(chunks_created=n)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest) -> RetrieveResponse:
    hits = retrieve(req.query, top_k=req.top_k, tickers=req.tickers)
    return RetrieveResponse(hits=[RetrieveHit(**h) for h in hits])


STATIC_DIR = str(__import__("pathlib").Path(__file__).parent / "static")

@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR + "/index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



@app.post("/analyze", response_model=AnalysisResponse)
def analyze_endpoint(req: AnalysisRequest) -> AnalysisResponse:
    result = analyze(req.query, top_k=req.top_k, tickers=req.tickers)
    citations = [Citation(**c) for c in result.get("citations", []) if isinstance(c, dict)]
    return AnalysisResponse(
        summary=result.get("summary", ""),
        key_points=result.get("key_points", []),
        risks=result.get("risks", []),
        citations=citations,
    )



@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(req: AgentRequest) -> AgentResponse:
    result = run_agent(req.query)
    trace = [AgentStep(**s) for s in result.get("trace", [])]
    return AgentResponse(answer=result.get("answer", ""), trace=trace)
