import logging

import yfinance as yf

log = logging.getLogger("gnomefinance.tools.market")


def get_price(ticker: str, period: str = "1mo") -> dict:
    """Get recent price history for a stock ticker."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        return {"error": f"yfinance failed: {e}"}
    if hist.empty:
        return {"error": f"No data for {ticker}"}
    close = hist["Close"]
    return {
        "ticker": ticker.upper(),
        "period": period,
        "latest": round(float(close.iloc[-1]), 2),
        "change_pct": round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2),
        "points": {str(k.date()): round(float(v), 2) for k, v in close.tail(30).items()},
    }


def compute_indicator(ticker: str, indicator: str = "rsi", period: int = 14) -> dict:
    """Compute RSI, SMA, or EMA for a ticker."""
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
    except Exception as e:
        return {"error": f"yfinance failed: {e}"}
    if hist.empty:
        return {"error": f"No data for {ticker}"}
    close = hist["Close"]
    ind = indicator.lower()
    if ind == "rsi":
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        value = 100 - (100 / (1 + rs))
    elif ind == "sma":
        value = close.rolling(period).mean()
    elif ind == "ema":
        value = close.ewm(span=period).mean()
    else:
        return {"error": f"Unknown indicator: {indicator}. Use rsi, sma, or ema."}
    return {
        "ticker": ticker.upper(),
        "indicator": ind.upper(),
        "period": period,
        "value": round(float(value.iloc[-1]), 2),
    }
