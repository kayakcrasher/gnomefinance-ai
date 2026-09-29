from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    db: str


class IngestTextRequest(BaseModel):
    source: str
    content: str
    ticker: str | None = None


class IngestUrlRequest(BaseModel):
    url: str
    ticker: str | None = None


class IngestEdgarRequest(BaseModel):
    ticker: str
    form: str = "10-K"
    limit: int = 1


class IngestResponse(BaseModel):
    chunks_created: int


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    tickers: list[str] | None = None


class RetrieveHit(BaseModel):
    score: float
    content: str
    source: str
    ticker: str | None = None
    chunk_index: int


class RetrieveResponse(BaseModel):
    hits: list[RetrieveHit]



class AnalysisRequest(BaseModel):
    query: str
    top_k: int = 5
    tickers: list[str] | None = None


class Citation(BaseModel):
    chunk_index: int
    snippet: str


class AnalysisResponse(BaseModel):
    summary: str
    key_points: list[str]
    risks: list[str]
    citations: list[Citation]
