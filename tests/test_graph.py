"""Offline tests for the full research graph, using a fake LLM and fake data.

No network, no API key, no cost: run them as often as you like with `pytest`.

The fake LLM behaves just enough like Claude to exercise every path:
- An analyst's first call asks for a tool, so the ToolNode loop really runs.
- Once a tool result is in the conversation, it writes a "report" that quotes
  that result, so we can check the data reached the report.
- The research manager gets a valid structured `InvestmentDecision`.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

import src.data as data
from src.graph import build_graph, make_debate_router
from src.state import InvestmentDecision

ANALYSTS = {"market_analyst", "fundamentals_analyst", "news_analyst", "sentiment_analyst"}
START_STATE = {"ticker": "FAKE", "as_of_date": "2026-01-02"}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeClaude(BaseChatModel):
    """A tiny scripted stand-in for ChatAnthropic.

    It decides what to say from the messages alone (no shared counters), so it
    is safe to use from the four analysts running in parallel.
    """

    tool_names: list[str] = []
    tools_disabled: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake-claude"

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        return FakeClaude(
            tool_names=[t.name for t in tools],
            tools_disabled=(tool_choice == {"type": "none"}),
        )

    def with_structured_output(self, schema, **kwargs):
        decision = schema(
            rating="HOLD",
            confidence=60,
            time_horizon="3-6 months",
            key_reasons=["fake reason"],
            key_risks=["fake risk"],
            strongest_counterargument="fake counter-argument",
            what_would_change_my_mind="fake trigger",
            investment_plan="## Recommendation\n\nHOLD (fake)",
        )
        return RunnableLambda(lambda _messages: decision)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if self.tool_names and not tool_results and not self.tools_disabled:
            # First analyst turn: ask for the first available tool.
            message = AIMessage(
                content="",
                tool_calls=[{"name": self.tool_names[0], "args": {"ticker": "FAKE"}, "id": "call_1"}],
            )
        elif tool_results:
            # Analyst has data: write a report that quotes it.
            message = AIMessage(content=f"Report based on: {tool_results[-1].content}\nVerdict: Neutral - fake")
        else:
            # Bull / bear turn.
            message = AIMessage(content="A fake debate argument.")
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture(autouse=True)
def fake_data(monkeypatch):
    """Replace every data function so no test ever contacts Yahoo Finance."""
    monkeypatch.setattr(data, "get_market_data", lambda t: f"FAKE MARKET DATA for {t}")
    monkeypatch.setattr(data, "get_fundamentals", lambda t: f"FAKE FUNDAMENTALS for {t}")
    monkeypatch.setattr(data, "get_news", lambda t, limit=20: f"FAKE NEWS for {t}")
    monkeypatch.setattr(data, "get_sentiment_data", lambda t: f"FAKE SENTIMENT for {t}")


def build_fake_graph(**kwargs):
    return build_graph(quick_llm=FakeClaude(), deep_llm=FakeClaude(), **kwargs)


def node_order(graph) -> list[str]:
    """Run the graph and return node names in the order they finished."""
    return [
        node_name
        for update in graph.stream(START_STATE, stream_mode="updates")
        for node_name in update
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_node_order_analysts_then_two_debate_rounds_then_manager():
    order = node_order(build_fake_graph(max_debate_rounds=2))

    # The 4 analysts finish first, in any order (they run in parallel).
    assert set(order[:4]) == ANALYSTS
    # Then exactly: bull, bear, bull, bear, manager.
    assert order[4:] == [
        "bull_researcher", "bear_researcher",
        "bull_researcher", "bear_researcher",
        "research_manager",
    ]


def test_bull_runs_once_after_the_analysts_join():
    """Fan-in check: the bull must not fire once per analyst."""
    order = node_order(build_fake_graph(max_debate_rounds=2))
    assert "bull_researcher" not in order[:4]
    assert order[4] == "bull_researcher"
    assert order.count("bull_researcher") == 2  # one per debate round, not 4 + 2


def test_debate_waits_for_all_four_analysts():
    """The bull is wired with a true join (one edge from the list of analysts).

    Why check the wiring, not just the run order? LangGraph runs in
    "supersteps": nodes triggered in the same step run together, and a node
    triggered several times in one step runs once. Our 4 analysts always
    finish in the same step, so even 4 separate edges would run the bull once
    today. The join only makes a difference when branches take different
    numbers of steps (e.g. if an analyst became a multi-node subgraph), and
    this test protects against that regression.
    """
    graph = build_fake_graph()
    assert (tuple(sorted(ANALYSTS)), "bull_researcher") in {
        (tuple(sorted(sources)), target) for sources, target in graph.builder.waiting_edges
    }


def test_final_state_is_complete():
    final = build_fake_graph(max_debate_rounds=2).invoke(START_STATE)

    assert final["debate_count"] == 4
    assert final["debate_history"].count("### Bull") == 2
    assert final["debate_history"].count("### Bear") == 2
    # Each analyst really called its tool and used the result.
    assert "FAKE MARKET DATA" in final["market_report"]
    assert "FAKE FUNDAMENTALS" in final["fundamentals_report"]
    assert "FAKE NEWS" in final["news_report"]
    assert "FAKE SENTIMENT" in final["sentiment_report"]
    # The manager's structured decision is valid and matches the plan.
    decision = InvestmentDecision(**final["decision"])
    assert decision.rating in {"BUY", "HOLD", "SELL"}
    assert final["investment_plan"] == decision.investment_plan


def test_debate_rounds_are_configurable():
    order = node_order(build_fake_graph(max_debate_rounds=1))
    assert order[4:] == ["bull_researcher", "bear_researcher", "research_manager"]


def test_failed_data_source_does_not_crash_the_run(monkeypatch):
    def broken_news(ticker, limit=20):
        raise data.DataUnavailableError("simulated Yahoo rate limit")

    monkeypatch.setattr(data, "get_news", broken_news)
    final = build_fake_graph().invoke(START_STATE)

    assert "DATA UNAVAILABLE" in final["news_report"]
    assert "FAKE MARKET DATA" in final["market_report"]  # other analysts unaffected
    assert final["decision"]["rating"] in {"BUY", "HOLD", "SELL"}


@pytest.mark.parametrize(
    ("debate_count", "expected"),
    [(0, "bull_researcher"), (2, "bull_researcher"), (3, "bull_researcher"), (4, "research_manager")],
)
def test_debate_router(debate_count, expected):
    route = make_debate_router(max_rounds=2)
    assert route({"debate_count": debate_count}) == expected
