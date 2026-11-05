# Decisions Log

A running record of the design choices behind this project, added phase by phase.
If you're picking this project up fresh, read this first.

---

## Phase 0: Kickoff and discovery (2026-11-05)

### Goals
| Question | Decision |
|---|---|
| Main goal | **Both**: learn LangGraph *and* use it for real research |
| Investment horizon | **Medium term** (a few months) |
| Test tickers | **NVDA, AAPL, WMT** (growth, mega-cap tech, defensive retail) |
| Universe | **US stocks only** (no ETFs or international tickers in v1) |

### Setup
| Question | Decision |
|---|---|
| Project folder | `~/Desktop/Invest Agent` (the GitHub repo name is chosen in Phase 9) |
| Environment | `venv` at `.venv/`, built from Anaconda Python 3.13.9 (`/opt/anaconda3/bin/python`). Never `/usr/bin/python3` (3.9.6, too old) |
| Models | **Quick:** `claude-haiku-5-5` (4 analysts + bull/bear researchers). **Deep:** `claude-sonnet-5-5` (research manager) |
| Cost ceiling | **Under $0.25 per ticker** |

### Model facts (checked in Anthropic docs on 2026-11-05)
- `claude-haiku-5-5`: $0.10 input / $0.50 output per million tokens (prompts ≤100k tokens; above that $0.50 / $2.50).
- `claude-sonnet-5-5`: $2 input / $10 output per million tokens.
- Rough estimate per ticker: 8 Haiku calls + 1–2 Sonnet calls ≈ **$0.10–0.15**.

---

## Phase 1: Project and environment setup (2026-11-05)

| Question | Decision |
|---|---|
| Git | git 2.39.5 installed; repo initialized on branch `main`. Commit identity: `nomunbileg <bilguuns38@gmail.com>` (the GitHub account email; user chose it over the noreply address) |
| LangSmith tracing | **Yes**. `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=invest-agents` in `.env` |

### Environment
- `.venv/` created from `/opt/anaconda3/bin/python` → Python 3.13.9.
- Pinned direct dependencies in `requirements.txt` (langgraph 1.2.20, langchain-anthropic 1.7.10, langchain-core 1.7.6, langsmith 0.14.17, yfinance 1.8.0, pandas 3.1.1, python-dotenv 1.2.4, pytest 9.2.0).
- `.gitignore` was created **before** `.env`. `.env` is a copy of `.env.example` that the user fills in themselves.
- Verified: both keys load from `.env` in the right format. One test call to `claude-haiku-5-5` replied "OK" (22 tokens, ≈$0.000004), and its trace appeared in the LangSmith project `invest-agents`.
- `instructions.md` (the build brief) stays local and uncommitted for now; to be revisited in the Phase 9 audit.

---

## Phase 2: Data layer (2026-11-05)

| Question | Decision |
|---|---|
| Data sources | **yfinance only** for v1. One fetch function per analyst, so sources can be swapped later |
| Price history and technicals | **1 year daily**; SMA 50/200, RSI 14, MACD 12/26/9, 1M/3M/1Y returns, 52-week high/low, last ~15 trading days as a table |
| News | **Up to 20 most recent** headlines (date, title, source, summary) |
| Sentiment | **Proxies** instead of social data: price vs. analyst targets, recommendation key, analyst count, short % of float, rating trend, recent headline tone |

### Implementation notes
- yfinance 1.8: `.news` returns only 10 items, so `get_news` uses `get_news(count=20)`. News arrives in the nested `content` format; flat-key parsing is kept for older versions.
- Invalid tickers don't raise in yfinance (they return empty history or a near-empty `.info`), so `data.py` raises its own `DataUnavailableError`.
- yfinance reports debt/equity as a percentage; it's converted to a ratio (16.97 → 0.17x).
- Yahoo's per-ticker news feed includes loosely related stories, and its ticker tags are too loose to filter on. The news and sentiment analyst prompts (Phase 4) will tell them to ignore headlines unrelated to the company.
- Output size per ticker: about 2,200 tokens across all four blocks (news is the largest at ~1,250).
