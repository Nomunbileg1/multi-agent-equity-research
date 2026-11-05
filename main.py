"""Command-line runner for the multi-agent investment research system.

Usage:
    python main.py NVDA
    python main.py NVDA AAPL WMT          # several tickers + a summary table
    python main.py NVDA --rounds 1        # shorter debate

For each ticker it streams the graph, prints a checkmark as each agent
finishes, then saves reports/<TICKER>_<YYYY-MM-DD>.md and .json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback

from src import config
from src.graph import build_graph, graph as default_graph
from src.report import build_json_summary, estimate_cost, render_markdown

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z.\-]{0,9}$")  # e.g. NVDA, BRK.B, BF-B


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research US stocks with a team of Claude agents (analysts, bull/bear debate, manager)."
    )
    parser.add_argument("tickers", nargs="+", help="one or more US stock tickers, e.g. NVDA AAPL WMT")
    parser.add_argument(
        "--rounds", type=int, default=config.MAX_DEBATE_ROUNDS,
        help=f"bull/bear debate rounds (default: {config.MAX_DEBATE_ROUNDS})",
    )
    parser.add_argument("--out", default="reports", help="output folder (default: reports)")
    return parser.parse_args()


def run_ticker(ticker: str, research_graph, out_dir: Path) -> dict:
    """Run the full graph for one ticker, save the reports and return a summary row."""
    as_of = date.today().isoformat()
    state: dict = {"ticker": ticker, "as_of_date": as_of}
    print(f"\n▶ Researching {ticker} (as of {as_of})")

    start = time.time()
    # The callback adds up token usage from every Claude call made inside this block.
    with get_usage_metadata_callback() as usage:
        # stream_mode="updates" yields {node_name: what_that_node_returned} as each node finishes.
        for update in research_graph.stream(state, stream_mode="updates"):
            for node_name, node_output in update.items():
                state.update(node_output)  # rebuild the final state as we go
                print(f"  ✓ {node_name} finished ({time.time() - start:.0f}s)")

    usage_by_model = dict(usage.usage_metadata)
    cost = estimate_cost(usage_by_model)

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{ticker}_{as_of}.md"
    json_path = out_dir / f"{ticker}_{as_of}.json"
    md_path.write_text(render_markdown(state, cost), encoding="utf-8")
    json_path.write_text(
        json.dumps(build_json_summary(state, usage_by_model, cost, str(md_path)), indent=2),
        encoding="utf-8",
    )

    decision = state.get("decision") or {}
    print_decision(ticker, decision, cost, time.time() - start, md_path)
    return {
        "ticker": ticker,
        "rating": decision.get("rating", "UNAVAILABLE"),
        "confidence": decision.get("confidence"),
        "cost": cost,
        "path": str(md_path),
    }


def print_decision(ticker: str, decision: dict, cost: float, seconds: float, path: Path) -> None:
    rating = decision.get("rating", "UNAVAILABLE")
    print(f"\n  {ticker}: {rating} (confidence {decision.get('confidence', 'n/a')}/100, "
          f"{decision.get('time_horizon', 'n/a')})")
    for reason in decision.get("key_reasons", []):
        print(f"    + {reason}")
    for risk in decision.get("key_risks", []):
        print(f"    - {risk}")
    if "error" in decision:
        print(f"    ! {decision['error']}")
    print(f"  Cost ≈ ${cost:.3f} · {seconds:.0f}s · Report: {path}")


def print_summary(rows: list[dict]) -> None:
    print("\nSummary")
    print(f"  {'Ticker':<8}{'Rating':<13}{'Conf.':<7}{'Cost':<9}Report")
    for row in rows:
        conf = "" if row["confidence"] is None else str(row["confidence"])
        cost = "" if row["cost"] is None else f"${row['cost']:.3f}"
        print(f"  {row['ticker']:<8}{row['rating']:<13}{conf:<7}{cost:<9}{row['path']}")
    total = sum(row["cost"] or 0 for row in rows)
    print(f"  Total cost ≈ ${total:.3f}")


def main() -> int:
    args = parse_args()
    research_graph = (
        default_graph if args.rounds == config.MAX_DEBATE_ROUNDS
        else build_graph(max_debate_rounds=args.rounds)
    )
    out_dir = Path(args.out)

    rows = []
    for raw in args.tickers:
        ticker = raw.strip().upper()
        if not TICKER_PATTERN.match(ticker):
            print(f"\n✗ Skipping '{raw}': not a valid ticker symbol")
            rows.append({"ticker": raw, "rating": "INVALID", "confidence": None, "cost": None, "path": "-"})
            continue
        try:
            rows.append(run_ticker(ticker, research_graph, out_dir))
        except Exception as error:  # noqa: BLE001  (one bad ticker shouldn't stop the others)
            print(f"\n✗ {ticker} failed: {type(error).__name__}: {error}")
            rows.append({"ticker": ticker, "rating": "FAILED", "confidence": None, "cost": None, "path": "-"})

    if len(rows) > 1:
        print_summary(rows)
    print("\nNot financial advice. Educational project.")
    return 0 if any(row["rating"] in {"BUY", "HOLD", "SELL"} for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
