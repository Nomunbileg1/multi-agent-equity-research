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

---

## Phase 3: State, config and LLM setup (2026-11-05)

| Question | Decision |
|---|---|
| Final decision format | **Both**: a written plan *and* a structured `InvestmentDecision` (rating, confidence 0–100, time_horizon, key_reasons, key_risks, strongest_counterargument, what_would_change_my_mind) |

### Design notes
- **One manager call, not two.** The written plan is a field (`investment_plan`) inside the structured `InvestmentDecision`, so Sonnet is called once. That's cheaper, and the rating and the text can't disagree.
- Structured output uses `with_structured_output(InvestmentDecision, method="json_schema")` (Claude's native structured outputs). The default `function_calling` method forces a tool call, which the API rejects while thinking is on.
- Claude 5.5 models think adaptively. `output_config={"effort": ...}` controls how much; config uses the API defaults (Haiku `medium`, Sonnet `high`). `max_tokens` includes thinking, so it's set with headroom (4000 / 8000).
- `decision` is stored in state as a plain dict (`model_dump()`) so it saves straight to JSON.
- Rating scale defaults to BUY/HOLD/SELL; Phase 5 confirms it.
- `src/llm.py` imports without an API key (the client is created lazily), so the offline tests can run in CI with no secrets.

---

## Phase 4: Analyst team (2026-11-05)

| Question | Decision |
|---|---|
| Analyst design | **(b) Tool-calling agents** in v1 (the brief's Phase 10 upgrade #1 moved up) |
| Agent implementation | **Hand-built LangGraph loop**: `llm ⇄ ToolNode` with `tools_condition`, one small compiled graph per analyst. (`create_react_agent` is deprecated in LangGraph 1.x; its replacement `langchain.agents.create_agent` was not used, to keep the loop visible and avoid an extra dependency) |
| Tools per analyst | **Own tool + related**. Market: market data. Fundamentals: fundamentals + market data. News: news + market data. Sentiment: sentiment + news |
| Tool-call cap | **Max 3 rounds** per analyst (`MAX_TOOL_ROUNDS`), then the LLM is called with `tool_choice={"type": "none"}` so it must write |
| Report style | **Under 400 words**: summary, bullets with numbers, `Factor / Observation / Signal` table, final `Verdict:` line |
| Prompts | Draft prompts accepted as shown; "use only data provided" became "use only data returned by your tools" |

### Design notes
- New module `src/tools.py` wraps the `data.py` functions as LangChain `@tool`s. Tools never raise; failures return `DATA UNAVAILABLE: ...` so the analyst can report the gap.
- Each analyst node runs its private agent loop and returns only `{<name>_report: text}`, so the outer graph's parallel fan-out is unchanged.
- Any exception inside an analyst (API outage, empty reply) becomes a "data unavailable" report with a Neutral verdict instead of crashing the run.
- `make_analyst(..., llm=...)` accepts an injected model so offline tests can use a fake LLM.

### Live verification (NVDA, 2026-11-05)
- All 4 analysts ran live: 1–3 tool calls each, every report ended with a verdict. **Total cost: $0.0053** (Haiku only).
- Verdicts: Market **Bearish**, Fundamentals **Bullish**, News **Neutral**, Sentiment **Bullish**.
- Fixes from that run:
  - **FCF data bug:** Yahoo's `info["freeCashflow"]` reported $41.8B for NVDA against $127.0B of reported FCF over the last 4 quarters, which led the analyst to a false "weak cash conversion" finding. `get_fundamentals` now sums operating cash flow, capex and FCF from the last 4 quarterly cash-flow statements, using the info field only as a labeled fallback.
  - Prompt: "never repeat a tool call with the same arguments" (the news analyst had fetched news twice).
  - Prompt: the word limit explicitly includes the table (two reports ran to 423 and 456 words).
