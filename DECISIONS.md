# Decisions Log

A running record of the design choices behind this project, added phase by phase.
If you're picking this project up fresh, read this first.

---

## Phase 0: Kickoff and discovery

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
| Project folder | `~/Desktop/Invest Agent` |
| Environment | `venv` at `.venv/`, built from Anaconda Python 3.13.9 (`/opt/anaconda3/bin/python`). Never `/usr/bin/python3` (3.9.6, too old) |
| Models | **Quick:** `claude-haiku-5-5` (4 analysts + bull/bear researchers). **Deep:** `claude-sonnet-5-5` (research manager) |
| Cost ceiling | **Under $0.25 per ticker** |

### Model facts (checked in Anthropic docs)
- `claude-haiku-5-5`: $0.10 input / $0.50 output per million tokens (prompts ≤100k tokens; above that $0.50 / $2.50).
- `claude-sonnet-5-5`: $2 input / $10 output per million tokens.
- Rough estimate per ticker: 8 Haiku calls + 1–2 Sonnet calls ≈ **$0.10–0.15**.

---

## Phase 1: Project and environment setup

| Question | Decision |
|---|---|
| Git | git 2.39.5 installed; repo initialized on branch `main`. Commit identity: `nomunbileg <bilguuns38@gmail.com>` (the GitHub account email; user chose it over the noreply address) |
| LangSmith tracing | **Yes**. `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=invest-agents` in `.env` |

### Environment
- `.venv/` created from `/opt/anaconda3/bin/python` → Python 3.13.9.
- Pinned direct dependencies in `requirements.txt` (langgraph 1.2.20, langchain-anthropic 1.7.10, langchain-core 1.7.6, langsmith 0.14.17, yfinance 1.8.0, pandas 3.1.1, python-dotenv 1.2.4, pytest 9.2.0).
- `.gitignore` was created **before** `.env`. `.env` is a copy of `.env.example` that the user fills in themselves.
- Verified: both keys load from `.env` in the right format. One test call to `claude-haiku-5-5` replied "OK" (22 tokens, ≈$0.000004), and its trace appeared in the LangSmith project `invest-agents`.
- `instructions.md` (the build brief) stays local and is never committed.

---

## Phase 2: Data layer

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

## Phase 3: State, config and LLM setup

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

## Phase 4: Analyst team

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

### Live verification (NVDA)
- All 4 analysts ran live: 1–3 tool calls each, every report ended with a verdict. **Total cost: $0.0053** (Haiku only).
- Verdicts: Market **Bearish**, Fundamentals **Bullish**, News **Neutral**, Sentiment **Bullish**.
- Fixes from that run:
  - **FCF data bug:** Yahoo's `info["freeCashflow"]` reported $41.8B for NVDA against $127.0B of reported FCF over the last 4 quarters, which led the analyst to a false "weak cash conversion" finding. `get_fundamentals` now sums operating cash flow, capex and FCF from the last 4 quarterly cash-flow statements, using the info field only as a labeled fallback.
  - Prompt: "never repeat a tool call with the same arguments" (the news analyst had fetched news twice).
  - Prompt: the word limit explicitly includes the table (two reports ran to 423 and 456 words).

---

## Phase 5: Researcher team

| Question | Decision |
|---|---|
| Debate rounds | **2** (`MAX_DEBATE_ROUNDS`): bull → bear → bull → bear |
| Concessions | **Yes**. A side may concede clearly correct points but must explain why its thesis still holds |
| Rating scale | **BUY / HOLD / SELL** (nuance carried by `confidence` 0–100) |
| Prompts | Draft bull, bear and manager prompts accepted as shown |

### Design notes
- `make_researcher(side, opponent)` builds both debaters from one template. Each turn appends `### Bull (turn N)` to `debate_history`, sets `{side}_last`, and increments `debate_count`.
- The research manager is a single Sonnet call with `with_structured_output(InvestmentDecision, method="json_schema")`. The written plan is the `investment_plan` field (sections: Recommendation, Rationale, Strongest Counter-Argument, Key Risks, What Would Change My Mind).
- `DEEP_MAX_TOKENS` was raised from 8000 to 12000, because a response cut off by the token limit would break the JSON. Worst-case cost per ticker is still under $0.25.
- Failures degrade gracefully: a failed debate turn records "(could not respond)" and still increments the count, so the loop always ends; a failed manager returns `rating: "UNAVAILABLE"` with the error.

---

## Phase 6: Graph wiring

No questions this phase (none in the brief).

### Design notes
- `build_graph(quick_llm, deep_llm, max_debate_rounds)` builds the graph, so tests can pass fake models. The module-level `graph = build_graph()` uses real Claude and is what `main.py` and LangGraph Studio load.
- Fan-out: `START → each analyst`. Fan-in: one edge from the **list** of analysts to `bull_researcher`. Loop: a conditional edge after `bear_researcher` goes back to `bull_researcher` while `debate_count < 2 * MAX_DEBATE_ROUNDS`, otherwise on to `research_manager`.
- **Tests use a custom `FakeClaude` (a small `BaseChatModel`) instead of `FakeListChatModel`.** `FakeListChatModel` can't `bind_tools`, so it couldn't exercise the tool-calling analysts. `FakeClaude` asks for a tool on its first analyst turn, writes a report quoting the tool result, and returns a valid `InvestmentDecision`. It decides from the messages alone, so it's safe across the 4 parallel analysts.
- **Superstep finding:** with all 4 analysts finishing in the same LangGraph superstep, four separate edges would *also* run the bull once, so a run-order test alone can't catch a broken join. `test_debate_waits_for_all_four_analysts` checks the join edge directly. A mutation check confirmed it fails when the join is replaced with separate edges.
- `tests/conftest.py` turns LangSmith tracing off and needs no API key, so CI can run the tests with no secrets.
- Diagram: Mermaid saved to `docs/graph.mmd`. The PNG export (`draw_mermaid_png`) was skipped because it uploads the diagram to mermaid.ink. GitHub renders the Mermaid natively, and LangGraph Studio screenshots show the graph visually.

---

## Phase 7: Runner and output

| Question | Decision |
|---|---|
| Output format | **Markdown + JSON**: `reports/<TICKER>_<YYYY-MM-DD>.md` and `.json` (decision, models, token usage, estimated cost) |
| Multiple tickers | **Yes**: `python main.py NVDA AAPL WMT` runs them in turn and prints a summary table. One failed ticker doesn't stop the others |
| Terminal output | **Checkmarks + decision**: `✓ <node> finished (Ns)`, then rating, confidence, reasons, risks, cost and report path |

### Design notes
- `main.py` streams with `stream_mode="updates"` and rebuilds the final state from the updates. Token usage is collected with `get_usage_metadata_callback()` and priced from `config.MODEL_PRICES`.
- Rendering lives in `src/report.py` (testable offline). The report opens with a "Decision at a Glance" table and ends with a one-line disclaimer.
- Extras: `--rounds` overrides the debate length; `--out` changes the output folder; ticker input is validated before any API call.
- Tests: `tests/test_report.py` checks the disclaimer, sections, heading nesting, cost math, and an end-to-end `run_ticker` with the fake graph writing both files. 15 offline tests in total.

### Live verification (NVDA)
- `python main.py NVDA` ran end to end in **70s for $0.051** (Haiku 37.8k in / 13.7k out tokens; Sonnet 7.4k in / 2.5k out). Decision: **BUY, confidence 58/100, 3–6 months**.
- Sonnet's structured output worked first time with `method="json_schema"`. The FCF fix shows correctly ($127.0B), and analyst reports stayed near 400 words.
- Debate quality check: the bull wrongly said the 200-day SMA hadn't been lost. The bear caught it, and the bull conceded on the next turn. The "data-center curbs" risk in the decision traces back to the news analyst's report (not invented).
- Fix: an analyst began its report with its own `## Title`, which looked like a new top-level report section. `report.py` now nests `#`/`##` headings inside all agent-written content under `###`, with a regression test (16 offline tests in total).

### Extra live tests
- **NFGC** (New Found Gold, small-cap gold developer): **SELL 60/100**, $0.047, 62s. Thin data handled honestly: the single-analyst $5 target was called "one analyst's view, not a consensus"; P/E and PEG were marked unusable; the cited milestones (Jan EPCM contract, Feb Keats results) match real Yahoo news items.
- **DRY.V** (Dryden Gold, TSX Venture explorer, outside the v1 US-only scope): **SELL 68/100**, $0.040, 56s. The free pre-check found two data-layer bugs, fixed before the live run:
  - **Currency was unlabeled:** CAD prices printed as "$". Market data now states `All prices are in <currency>` (from `get_history_metadata()`, no extra request); fundamentals and sentiment show `info["currency"]`. The agents then wrote "C$" throughout (25 mentions, 0 USD slips).
  - **Penny-stock precision:** 0.175 rendered as 0.17. A new `_price()` helper uses 4 decimals below $1 (prices, SMAs, MACD, targets).
  - Claims verified against raw data: FCF -C$13.09M, the +20.69% move on 11/04 on 855K shares, cash and debt n/a ("unverified").
- Takeaway: the pipeline works for small caps and Canadian listings, but Yahoo's coverage there is thin (4–12 news items, 1 analyst), so calls rest on less evidence. That's a stronger case for the Phase 10 "better data" upgrade.

