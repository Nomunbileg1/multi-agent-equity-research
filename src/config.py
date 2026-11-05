"""Central configuration: every tunable setting lives here.

Change a value here and the whole system picks it up. Phase 3 adds model and
debate settings below the data settings.
"""

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
