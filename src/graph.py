"""LangGraph wiring: connect the analysts, the debate and the manager.

    START ─┬─ market_analyst ───────┐
           ├─ fundamentals_analyst ─┤
           ├─ news_analyst ─────────┼─► bull_researcher ⇄ bear_researcher ─► research_manager ─► END
           └─ sentiment_analyst ────┘          (MAX_DEBATE_ROUNDS rounds)

LangGraph concepts used here:
- FAN-OUT: four edges from START, one to each analyst. LangGraph runs all
  nodes that are ready in the same "step" in parallel, so the four analysts
  work at the same time.
- FAN-IN (join): one edge from a *list* of nodes to the bull researcher. The
  bull waits until every node in the list has finished, then runs once. (Four
  separate edges would instead trigger the bull once per analyst.)
- CONDITIONAL EDGE: after each bear turn, a small routing function reads the
  state and returns the name of the next node: back to the bull (loop) or on
  to the manager.

The compiled `graph` at the bottom is what `main.py`, the tests and
LangGraph Studio (via langgraph.json) all load.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src import config
from src.analysts import make_analyst_team
from src.researchers import make_research_manager, make_researcher
from src.state import ResearchState


def make_debate_router(max_rounds: int) -> Callable[[ResearchState], str]:
    """Return the routing function used after each bear turn.

    One round is one bull turn plus one bear turn, so the debate is over after
    `2 * max_rounds` turns.
    """

    def route_after_bear(state: ResearchState) -> str:
        if state.get("debate_count", 0) < 2 * max_rounds:
            return "bull_researcher"  # loop: another round
        return "research_manager"     # debate over

    return route_after_bear


def build_graph(
    quick_llm: BaseChatModel | None = None,
    deep_llm: BaseChatModel | None = None,
    max_debate_rounds: int = config.MAX_DEBATE_ROUNDS,
):
    """Build and compile the research graph.

    By default it uses the real Claude models from src/llm.py. Tests pass in
    fake models instead, so the whole graph can run offline for free.
    """
    builder = StateGraph(ResearchState)

    # --- Nodes -------------------------------------------------------------
    analysts = make_analyst_team(quick_llm)
    for name, node in analysts.items():
        builder.add_node(name, node)
    builder.add_node("bull_researcher", make_researcher("bull", "bear", quick_llm))
    builder.add_node("bear_researcher", make_researcher("bear", "bull", quick_llm))
    builder.add_node("research_manager", make_research_manager(deep_llm))

    # --- Edges -------------------------------------------------------------
    # Fan-out: every analyst starts at the same time.
    for name in analysts:
        builder.add_edge(START, name)

    # Fan-in: the debate starts only after ALL analysts have finished.
    builder.add_edge(list(analysts), "bull_researcher")

    # The debate: bull, then bear, then either loop or finish.
    builder.add_edge("bull_researcher", "bear_researcher")
    builder.add_conditional_edges(
        "bear_researcher",
        make_debate_router(max_debate_rounds),
        ["bull_researcher", "research_manager"],  # possible destinations (for the diagram)
    )
    builder.add_edge("research_manager", END)

    return builder.compile()


# Module-level compiled graph, loaded by main.py and LangGraph Studio.
graph = build_graph()
