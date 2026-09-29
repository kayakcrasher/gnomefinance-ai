import logging
import time

from google import genai
from google.genai import types

from app.config import settings
from app.tools.market import compute_indicator, get_price

log = logging.getLogger("gnomefinance.agent")

_client = None
MODEL_CHAIN = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash"]
MAX_STEPS = 5
MAX_RETRIES = 3
BACKOFF = 2.0


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


TOOL_IMPLS = {
    "get_price": get_price,
    "compute_indicator": compute_indicator,
}

TOOL_DECLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_price",
            description="Get recent price history for a stock ticker.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "ticker": types.Schema(type="STRING", description="Stock ticker, e.g. NVDA"),
                    "period": types.Schema(type="STRING", description="One of 5d, 1mo, 3mo, 6mo, 1y"),
                },
                required=["ticker"],
            ),
        ),
        types.FunctionDeclaration(
            name="compute_indicator",
            description="Compute a technical indicator for a ticker: rsi, sma, or ema.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "ticker": types.Schema(type="STRING"),
                    "indicator": types.Schema(type="STRING", description="rsi, sma, or ema"),
                    "period": types.Schema(type="INTEGER", description="Lookback period, default 14"),
                },
                required=["ticker", "indicator"],
            ),
        ),
    ])
]

SYSTEM = """You are a financial analyst agent. You have tools that fetch live market data.

When asked about a stock's price action or technicals, USE THE TOOLS rather than guessing.
After gathering data, summarize clearly. Be concise and factual."""


def _generate_with_retry(contents):
    client = _get_client()
    last_err = None
    for model in MODEL_CHAIN:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        tools=TOOL_DECLS,
                        temperature=0.2,
                    ),
                )
            except Exception as e:
                log.warning("model=%s attempt=%d failed: %s", model, attempt, e)
                last_err = e
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF * attempt)
    raise RuntimeError(f"All models failed. Last error: {last_err}")


def run_agent(query: str) -> dict:
    contents = [types.Content(role="user", parts=[types.Part(text=query)])]
    trace: list[dict] = []

    for _ in range(MAX_STEPS):
        try:
            resp = _generate_with_retry(contents)
        except Exception as e:
            return {"answer": f"Agent failed: {e}", "trace": trace}

        parts = resp.candidates[0].content.parts
        function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

        if not function_calls:
            text = "".join(p.text for p in parts if getattr(p, "text", None))
            return {"answer": text, "trace": trace}

        contents.append(resp.candidates[0].content)

        for fc in function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            impl = TOOL_IMPLS.get(name)
            if impl is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = impl(**args)
                except Exception as e:
                    result = {"error": str(e)}
            trace.append({"tool": name, "args": args, "result": result})
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=name, response=result)],
                )
            )

    return {"answer": "Agent exceeded max steps.", "trace": trace}
