"""Central configuration: every tunable setting lives here.

Change a value here and the whole system picks it up.
"""

# --- Models (IDs checked against Anthropic's model docs, 2026-11-05) --------

# "Quick" model: the 4 analysts and the bull/bear researchers (many calls, cheap).
QUICK_MODEL = "claude-haiku-5-5"
# "Deep" model: the research manager (one call, needs the best judgment).
DEEP_MODEL = "claude-sonnet-5-5"

# Max tokens per response. On Claude 5.5 models this includes any thinking tokens,
# so leave headroom above the visible report length.
QUICK_MAX_TOKENS = 4000
DEEP_MAX_TOKENS = 12000  # thinking + a ~600-word plan + JSON fields; a cut-off would break the JSON

# Effort ("low" | "medium" | "high" | "xhigh" | "max") controls how much the
# model thinks before answering. These match each model's API default.
QUICK_EFFORT = "medium"
DEEP_EFFORT = "high"

# USD per million tokens (input, output), used to estimate cost per run.
MODEL_PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),   # prompts up to 100k tokens
    "claude-sonnet-5-5": (2.00, 10.00),
}

# --- Analysts ----------------------------------------------------------------

# Max rounds of tool calls per analyst before it must write its report (caps cost).
MAX_TOOL_ROUNDS = 3

# --- Debate ------------------------------------------------------------------

# One round = one bull turn + one bear turn.
MAX_DEBATE_ROUNDS = 2

# --- Data settings (used by src/data.py) ------------------------------------

# Price history window for the market/technical analyst (a yfinance period string).
PRICE_LOOKBACK = "1y"

# How many of the most recent trading days to show as a raw price table.
RECENT_DAYS_TABLE = 15

# Technical indicator parameters.
SMA_SHORT = 50
SMA_LONG = 200
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

# Approximate trading days in each return window.
TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63

# News analyst: maximum number of recent headlines to fetch.
NEWS_LIMIT = 20

# Sentiment analyst: how many recent headlines to include for tone.
SENTIMENT_HEADLINES = 10

# Max characters kept from each news summary (keeps prompts short and cheap).
NEWS_SUMMARY_CHARS = 300
