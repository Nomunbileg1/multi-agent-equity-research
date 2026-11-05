"""Data layer: fetch market data from Yahoo Finance and format it as plain text.

Each analyst has one fetch function, and each returns a plain-text block for
the LLM to read:

    get_market_data(ticker)     -> market / technical analyst
    get_fundamentals(ticker)    -> fundamentals analyst
    get_news(ticker, limit)     -> news analyst
    get_sentiment_data(ticker)  -> sentiment analyst

If a ticker is invalid or Yahoo returns nothing, the function raises
`DataUnavailableError`. The analyst node catches it and reports "data
unavailable" instead of crashing the run. Any one missing field just shows
as "n/a".

To switch data providers later (e.g. Finnhub), replace a function's body and
keep its name and return type. Nothing else needs to change.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src import config

# yfinance prints HTTP errors (e.g. "404 Not Found") to the console. We handle
# failures ourselves, so we silence its logger to keep the output clean.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class DataUnavailableError(Exception):
    """Raised when no usable data can be fetched for a ticker."""


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def _to_float(value) -> float | None:
    """Convert numpy/pandas numbers to a plain float; return None for missing/NaN."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _num(value, decimals: int = 2) -> str:
    """Format a plain number, e.g. 22.958 -> '22.96'."""
    number = _to_float(value)
    return "n/a" if number is None else f"{number:,.{decimals}f}"


def _pct(value, decimals: int = 1) -> str:
    """Format a ratio as a percentage, e.g. 0.7467 -> '74.7%'."""
    number = _to_float(value)
    return "n/a" if number is None else f"{number * 100:.{decimals}f}%"


def _money(value) -> str:
    """Format a dollar amount with a readable suffix, e.g. 41809874944 -> '$41.81B'."""
    number = _to_float(value)
    if number is None:
        return "n/a"
    for divisor, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
        if abs(number) >= divisor:
            return f"${number / divisor:,.2f}{suffix}"
    return f"${number:,.2f}"


def _clean_ticker(ticker: str) -> str:
    """Normalize user input like ' nvda ' -> 'NVDA'."""
    return ticker.strip().upper()


def _get_info(ticker: str) -> dict:
    """Fetch the yfinance `.info` dict, raising DataUnavailableError for unknown tickers.

    yfinance doesn't raise for a bad symbol; it returns a nearly empty dict.
    A real stock always has a `quoteType` field, so we check for that.
    """
    info = yf.Ticker(ticker).info or {}
    if not info.get("quoteType"):
        raise DataUnavailableError(f"No quote data found for '{ticker}' (invalid or delisted ticker?)")
    return info


# ---------------------------------------------------------------------------
# 1. Market / technical data
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (the standard RSI formula)."""
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def _period_return(close: pd.Series, days: int) -> float | None:
    """Percentage return over the last `days` trading days (None if not enough data)."""
    if len(close) <= days:
        return None
    return _to_float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def get_market_data(ticker: str) -> str:
    """Price history, technical indicators and recent performance as plain text."""
    ticker = _clean_ticker(ticker)
    history = yf.Ticker(ticker).history(period=config.PRICE_LOOKBACK)
    if history.empty:
        raise DataUnavailableError(f"No price history found for '{ticker}'")

    close = history["Close"]

    # Indicators (each is a full series; we report the latest value).
    sma_short = close.rolling(config.SMA_SHORT).mean()
    sma_long = close.rolling(config.SMA_LONG).mean()
    rsi = _rsi(close, config.RSI_PERIOD)
    ema_fast = close.ewm(span=config.MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=config.MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=config.MACD_SIGNAL, adjust=False).mean()

    last_close = _to_float(close.iloc[-1])
    last_sma_short = _to_float(sma_short.iloc[-1])
    last_sma_long = _to_float(sma_long.iloc[-1])
    high_52w = _to_float(history["High"].max())
    low_52w = _to_float(history["Low"].min())
    last_date = history.index[-1].strftime("%Y-%m-%d")

    def vs(level: float | None) -> str:
        """Describe where the price sits relative to a level, e.g. 'above (+4.2%)'."""
        if level is None or last_close is None:
            return "n/a"
        diff = last_close / level - 1
        return f"{'above' if diff >= 0 else 'below'} ({diff * 100:+.1f}%)"

    if last_sma_short is not None and last_sma_long is not None:
        # This describes the current state, not a fresh crossover event.
        if last_sma_short > last_sma_long:
            cross = f"SMA {config.SMA_SHORT} is above SMA {config.SMA_LONG} (longer-term uptrend)"
        else:
            cross = f"SMA {config.SMA_SHORT} is below SMA {config.SMA_LONG} (longer-term downtrend)"
    else:
        cross = "n/a (not enough history)"

    lines = [
        f"MARKET DATA FOR {ticker} (last close {last_date}, {len(history)} trading days of history)",
        "",
        "Performance:",
        f"- Last close: ${_num(last_close)}",
        f"- 1-month return: {_pct(_period_return(close, config.TRADING_DAYS_1M))}",
        f"- 3-month return: {_pct(_period_return(close, config.TRADING_DAYS_3M))}",
        f"- 1-year return: {_pct(_period_return(close, len(close) - 1))}",
        f"- 52-week high: ${_num(high_52w)} (price is {vs(high_52w)} the high)",
        f"- 52-week low: ${_num(low_52w)} (price is {vs(low_52w)} the low)",
        "",
        "Technical indicators:",
        f"- SMA {config.SMA_SHORT}: ${_num(last_sma_short)} (price {vs(last_sma_short)})",
        f"- SMA {config.SMA_LONG}: ${_num(last_sma_long)} (price {vs(last_sma_long)})",
        f"- Moving-average trend: {cross}",
        f"- RSI {config.RSI_PERIOD}: {_num(rsi.iloc[-1], 1)} (above 70 = overbought, below 30 = oversold)",
        f"- MACD ({config.MACD_FAST}/{config.MACD_SLOW}/{config.MACD_SIGNAL}): "
        f"MACD {_num(macd.iloc[-1])}, signal {_num(macd_signal.iloc[-1])}, "
        f"histogram {_num(macd.iloc[-1] - macd_signal.iloc[-1])}",
        f"- Average daily volume (20d vs 50d): "
        f"{_num(history['Volume'].tail(20).mean(), 0)} vs {_num(history['Volume'].tail(50).mean(), 0)}",
        "",
        f"Last {config.RECENT_DAYS_TABLE} trading days:",
        "| Date | Close | Change | Volume |",
        "|---|---|---|---|",
    ]
    recent = history.tail(config.RECENT_DAYS_TABLE + 1)
    daily_change = recent["Close"].pct_change()
    for date, row in recent.iloc[1:].iterrows():
        lines.append(
            f"| {date.strftime('%Y-%m-%d')} | ${_num(row['Close'])} | "
            f"{_pct(daily_change.loc[date], 2)} | {_num(row['Volume'], 0)} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Fundamentals
# ---------------------------------------------------------------------------

# Rows we pull from the quarterly income statement, in display order.
_INCOME_ROWS = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "Diluted EPS"]


def get_fundamentals(ticker: str) -> str:
    """Valuation, growth, profitability, balance sheet and the last 4 quarters as plain text."""
    ticker = _clean_ticker(ticker)
    info = _get_info(ticker)

    # yfinance reports debt/equity as a percentage (16.97 means 0.17x).
    debt_to_equity = _to_float(info.get("debtToEquity"))
    debt_to_equity_text = "n/a" if debt_to_equity is None else f"{debt_to_equity / 100:.2f}x"
    peg = info.get("pegRatio") or info.get("trailingPegRatio")

    lines = [
        f"FUNDAMENTALS FOR {ticker}: {info.get('longName', ticker)}",
        f"Sector: {info.get('sector', 'n/a')} | Industry: {info.get('industry', 'n/a')} | "
        f"Market cap: {_money(info.get('marketCap'))} | Price: ${_num(info.get('currentPrice'))}",
        "",
        "Valuation:",
        f"- Trailing P/E: {_num(info.get('trailingPE'))}",
        f"- Forward P/E: {_num(info.get('forwardPE'))}",
        f"- Price/Book: {_num(info.get('priceToBook'))}",
        f"- PEG ratio: {_num(peg)}",
        "",
        "Growth (year over year, most recent quarter):",
        f"- Revenue growth: {_pct(info.get('revenueGrowth'))}",
        f"- Earnings growth: {_pct(info.get('earningsGrowth'))}",
        "",
        "Profitability:",
        f"- Gross margin: {_pct(info.get('grossMargins'))}",
        f"- Operating margin: {_pct(info.get('operatingMargins'))}",
        f"- Net profit margin: {_pct(info.get('profitMargins'))}",
        f"- Return on equity: {_pct(info.get('returnOnEquity'))}",
        "",
        "Balance sheet and cash flow:",
        f"- Debt/Equity: {debt_to_equity_text}",
        f"- Free cash flow (trailing 12m): {_money(info.get('freeCashflow'))}",
        f"- Total cash: {_money(info.get('totalCash'))}",
        f"- Total debt: {_money(info.get('totalDebt'))}",
        "",
    ]
    lines += _quarterly_table(ticker)
    return "\n".join(lines)


def _quarterly_table(ticker: str) -> list[str]:
    """The last 4 quarters of key income-statement lines as a markdown table."""
    try:
        statement = yf.Ticker(ticker).quarterly_income_stmt
    except Exception:  # noqa: BLE001  (any failure here just means "no table")
        statement = None
    if statement is None or statement.empty:
        return ["Last 4 quarters: data unavailable"]

    quarters = list(statement.columns[:4])  # newest first
    header = "| Metric | " + " | ".join(q.strftime("%Y-%m-%d") for q in quarters) + " |"
    lines = ["Last 4 quarters (newest first):", header, "|---" * (len(quarters) + 1) + "|"]
    for row_name in _INCOME_ROWS:
        if row_name not in statement.index:
            continue
        formatter = _num if row_name == "Diluted EPS" else _money
        cells = [formatter(statement.loc[row_name, q]) for q in quarters]
        lines.append(f"| {row_name} | " + " | ".join(cells) + " |")
    return lines


# ---------------------------------------------------------------------------
# 3. News
# ---------------------------------------------------------------------------

def _parse_news_item(item: dict) -> dict:
    """Normalize one yfinance news item into {date, title, source, summary}.

    yfinance changed its news format:
      - newer versions nest fields under item["content"] (title, summary,
        pubDate as an ISO string, provider.displayName)
      - older versions use flat keys (title, publisher, providerPublishTime
        as a Unix timestamp)
    This function handles both.
    """
    if "content" in item and isinstance(item["content"], dict):
        content = item["content"]
        date = (content.get("pubDate") or content.get("displayTime") or "")[:10]
        return {
            "date": date or "n/a",
            "title": content.get("title") or "(no title)",
            "source": (content.get("provider") or {}).get("displayName") or "n/a",
            "summary": content.get("summary") or content.get("description") or "",
        }

    timestamp = item.get("providerPublishTime")
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d") if timestamp else "n/a"
    return {
        "date": date,
        "title": item.get("title") or "(no title)",
        "source": item.get("publisher") or "n/a",
        "summary": item.get("summary") or "",
    }


def _fetch_news(ticker: str, limit: int) -> list[dict]:
    """Fetch up to `limit` recent news items, normalized and sorted newest first."""
    raw_items = yf.Ticker(ticker).get_news(count=limit) or []
    items = [_parse_news_item(item) for item in raw_items]
    items.sort(key=lambda item: item["date"], reverse=True)
    return items[:limit]


def get_news(ticker: str, limit: int = config.NEWS_LIMIT) -> str:
    """Recent headlines with date, source and a short summary, as plain text."""
    ticker = _clean_ticker(ticker)
    items = _fetch_news(ticker, limit)
    if not items:
        raise DataUnavailableError(f"No recent news found for '{ticker}'")

    lines = [f"RECENT NEWS FOR {ticker} ({len(items)} articles, newest first)", ""]
    for number, item in enumerate(items, start=1):
        summary = item["summary"].strip().replace("\n", " ")
        if len(summary) > config.NEWS_SUMMARY_CHARS:
            summary = summary[: config.NEWS_SUMMARY_CHARS].rsplit(" ", 1)[0] + "..."
        lines.append(f"{number}. [{item['date']}] {item['title']} ({item['source']})")
        if summary:
            lines.append(f"   {summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Sentiment (Wall Street ratings, price targets, short interest, headlines)
# ---------------------------------------------------------------------------

def get_sentiment_data(ticker: str) -> str:
    """Market-sentiment proxies as plain text.

    There's no social-media data in v1, so sentiment is measured from analyst
    price targets and ratings, the rating trend, short interest and the tone
    of recent headlines.
    """
    ticker = _clean_ticker(ticker)
    info = _get_info(ticker)
    price = _to_float(info.get("currentPrice"))

    def upside(target) -> str:
        """Percent move from the current price to a target, e.g. '+80.1%'."""
        target = _to_float(target)
        if target is None or price is None:
            return "n/a"
        return f"{(target / price - 1) * 100:+.1f}%"

    lines = [
        f"SENTIMENT DATA FOR {ticker}",
        "",
        "Analyst price targets:",
        f"- Current price: ${_num(price)}",
        f"- Mean target: ${_num(info.get('targetMeanPrice'))} ({upside(info.get('targetMeanPrice'))} vs. price)",
        f"- Median target: ${_num(info.get('targetMedianPrice'))} ({upside(info.get('targetMedianPrice'))})",
        f"- High target: ${_num(info.get('targetHighPrice'))} ({upside(info.get('targetHighPrice'))})",
        f"- Low target: ${_num(info.get('targetLowPrice'))} ({upside(info.get('targetLowPrice'))})",
        "",
        "Analyst consensus:",
        f"- Recommendation: {info.get('recommendationKey', 'n/a')} "
        f"(mean score {_num(info.get('recommendationMean'))} on a 1 = strong buy ... 5 = strong sell scale)",
        f"- Number of analysts: {info.get('numberOfAnalystOpinions', 'n/a')}",
        "",
        "Short interest:",
        f"- Short % of float: {_pct(info.get('shortPercentOfFloat'))}",
        f"- Short ratio (days to cover): {_num(info.get('shortRatio'))}",
        "",
    ]
    lines += _rating_trend(ticker)
    lines += ["", f"Recent headlines (for tone):"]
    try:
        headlines = _fetch_news(ticker, config.SENTIMENT_HEADLINES)
    except Exception:  # noqa: BLE001  (headlines are a bonus; don't fail the whole block)
        headlines = []
    if headlines:
        lines += [f"- [{h['date']}] {h['title']}" for h in headlines]
    else:
        lines.append("- none available")
    return "\n".join(lines)


def _rating_trend(ticker: str) -> list[str]:
    """Analyst rating counts over the last few months as a markdown table."""
    try:
        trend = yf.Ticker(ticker).recommendations
    except Exception:  # noqa: BLE001
        trend = None
    if trend is None or trend.empty:
        return ["Analyst rating trend: data unavailable"]

    lines = [
        "Analyst rating trend (number of analysts; 0m = this month):",
        "| Period | Strong Buy | Buy | Hold | Sell | Strong Sell |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in trend.iterrows():
        counts = [str(int(row.get(col, 0))) for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]]
        lines.append(f"| {row.get('period', 'n/a')} | " + " | ".join(counts) + " |")
    return lines
