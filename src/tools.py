"""LangChain tools that wrap the data layer so analysts can call them.

A "tool" is a Python function plus a name, description and argument schema that
Claude can see. Claude reads the docstring to decide *when* to call a tool, so
the docstrings below are written for the model, not just for humans.

Tools never raise: if a data source fails, the tool returns a short
"DATA UNAVAILABLE: ..." message instead, and the analyst reports that gap
rather than crashing the run.
"""

from typing import Callable

from langchain_core.tools import tool

from src import config, data


def _safe_call(fetch_fn: Callable[..., str], *args) -> str:
    """Run a data function and turn any failure into a readable message for the model."""
    try:
        return fetch_fn(*args)
    except data.DataUnavailableError as error:
        return f"DATA UNAVAILABLE: {error}"
    except Exception as error:  # noqa: BLE001  (rate limits, network errors, etc.)
        return f"DATA UNAVAILABLE: {type(error).__name__}: {error}"


@tool
def fetch_market_data(ticker: str) -> str:
    """Get price performance and technical indicators for a US stock.

    Returns 1M/3M/1Y returns, the 52-week high/low, SMA 50/200, RSI 14, MACD,
    average volume, and a table of the last 15 trading days.
    """
    return _safe_call(data.get_market_data, ticker)


@tool
def fetch_fundamentals(ticker: str) -> str:
    """Get company fundamentals for a US stock.

    Returns valuation (P/E, forward P/E, P/B, PEG), growth, margins, ROE,
    debt/equity, free cash flow, cash and debt, plus the last 4 quarters of
    revenue, gross profit, operating income, net income and EPS.
    """
    return _safe_call(data.get_fundamentals, ticker)


@tool
def fetch_news(ticker: str, limit: int = config.NEWS_LIMIT) -> str:
    """Get recent news headlines for a US stock, newest first.

    Each item has a date, title, source and short summary. Note: the feed can
    include stories that only mention the company in passing.
    """
    limit = max(1, min(limit, config.NEWS_LIMIT))  # keep the model within sensible bounds
    return _safe_call(data.get_news, ticker, limit)


@tool
def fetch_sentiment_data(ticker: str) -> str:
    """Get market-sentiment indicators for a US stock.

    Returns analyst price targets (mean/median/high/low and upside vs. price),
    consensus rating, number of analysts, short interest, the analyst rating
    trend over recent months, and recent headlines.
    """
    return _safe_call(data.get_sentiment_data, ticker)
