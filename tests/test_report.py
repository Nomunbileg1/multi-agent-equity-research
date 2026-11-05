"""Offline tests for report rendering and the CLI runner (fake LLM, no cost)."""

import json

import main
from src.graph import build_graph
from src.report import DISCLAIMER, estimate_cost, render_markdown
from tests.test_graph import FakeClaude, fake_data  # noqa: F401  (fake_data is an autouse fixture)

STATE = {
    "ticker": "FAKE",
    "as_of_date": "2026-01-02",
    "market_report": "market text",
    "debate_history": "### Bull (turn 1)\n\nargument",
    "investment_plan": "## Recommendation\n\nHOLD\n\n## Key Risks\n\nrisk",
    "decision": {"rating": "HOLD", "confidence": 60, "time_horizon": "3-6 months",
                 "key_reasons": ["r1"], "key_risks": ["k1"]},
}


def test_report_ends_with_disclaimer():
    assert render_markdown(STATE).rstrip().endswith(DISCLAIMER)


def test_report_has_all_sections_and_nests_plan_headings():
    md = render_markdown(STATE, cost_usd=0.1234)
    for heading in ["Market / Technical", "Fundamentals", "News", "Sentiment",
                    "Bull vs. Bear Debate", "Research Manager Decision"]:
        assert heading in md
    assert "### Recommendation" in md and "\n## Recommendation" not in md
    assert "$0.123" in md
    assert "Report unavailable." in md  # missing reports are labeled, not blank


def test_agent_written_titles_do_not_become_report_sections():
    """An analyst that starts with its own '## Title' must not break the report structure."""
    md = render_markdown({**STATE, "market_report": "## NVDA Technical Assessment\n\ntext",
                          "news_report": "# News Title\n\ntext"})
    top_level = [line for line in md.splitlines() if line.startswith(("# ", "## "))]
    assert top_level == [
        "# FAKE Investment Research Report", "## Decision at a Glance",
        "## 1. Market / Technical Analysis", "## 2. Fundamentals Analysis", "## 3. News Analysis",
        "## 4. Sentiment Analysis", "## 5. Bull vs. Bear Debate", "## 6. Research Manager Decision",
    ]
    assert "### NVDA Technical Assessment" in md and "### News Title" in md


def test_report_handles_unavailable_decision():
    md = render_markdown({"ticker": "FAKE", "as_of_date": "2026-01-02",
                          "decision": {"rating": "UNAVAILABLE", "error": "boom"}})
    assert "**UNAVAILABLE**" in md
    assert md.rstrip().endswith(DISCLAIMER)


def test_estimate_cost():
    usage = {
        "claude-haiku-5-5": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},   # $0.10 + $0.50
        "claude-sonnet-5-5": {"input_tokens": 100_000, "output_tokens": 10_000},       # $0.20 + $0.10
        "some-unknown-model": {"input_tokens": 999, "output_tokens": 999},             # ignored
    }
    assert abs(estimate_cost(usage) - 0.90) < 1e-9


def test_run_ticker_writes_markdown_and_json(tmp_path):
    fake_graph = build_graph(quick_llm=FakeClaude(), deep_llm=FakeClaude())
    row = main.run_ticker("FAKE", fake_graph, tmp_path)

    md_files, json_files = list(tmp_path.glob("FAKE_*.md")), list(tmp_path.glob("FAKE_*.json"))
    assert len(md_files) == 1 and len(json_files) == 1
    assert md_files[0].read_text().rstrip().endswith(DISCLAIMER)
    summary = json.loads(json_files[0].read_text())
    assert summary["decision"]["rating"] == "HOLD"
    assert summary["debate_rounds"] == 2
    assert row["rating"] == "HOLD"
