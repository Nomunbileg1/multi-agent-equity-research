"""Shared graph state and the structured decision schema.

LangGraph concept: STATE
------------------------
The state is one shared dictionary that flows through the graph. Every node
receives the current state and returns a *partial* dict with only the keys it
changed; LangGraph merges that update into the state. Because each node only
returns its own keys, the 4 analysts can run in parallel without overwriting
each other's reports.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class InvestmentDecision(BaseModel):
    """The research manager's final, machine-readable decision.

    The manager fills this in with Claude's structured output, so the written
    plan and the rating come from the same response and always agree.
    """

    rating: Literal["BUY", "HOLD", "SELL"] = Field(
        description="Final recommendation for a medium-term (several months) horizon."
    )
    confidence: int = Field(ge=0, le=100, description="Confidence in the rating, 0-100.")
    time_horizon: str = Field(description="Holding period the rating applies to, e.g. '3-6 months'.")
    key_reasons: list[str] = Field(description="2-4 most important reasons for the rating.")
    key_risks: list[str] = Field(description="2-4 most important risks to the thesis.")
    strongest_counterargument: str = Field(
        description="The strongest argument from the losing side of the debate."
    )
    what_would_change_my_mind: str = Field(
        description="Specific, observable developments that would flip the rating."
    )
    investment_plan: str = Field(
        description="The full written investment plan in markdown (the human-readable report)."
    )


class ResearchState(TypedDict, total=False):
    """Everything the graph knows about one ticker as it runs.

    `total=False` means no key is required up front: we start the graph with
    just `ticker` and `as_of_date`, and nodes fill in the rest.
    """

    # --- Input -------------------------------------------------------------
    ticker: str            # e.g. "NVDA"
    as_of_date: str        # "YYYY-MM-DD"; the date the analysis is run for

    # --- Analyst team (written in parallel, one key per analyst) -----------
    market_report: str
    fundamentals_report: str
    news_report: str
    sentiment_report: str

    # --- Researcher team (bull and bear take turns) ------------------------
    debate_history: str    # full transcript, appended to on every turn
    bull_last: str         # bull's most recent argument
    bear_last: str         # bear's most recent argument
    debate_count: int      # number of turns taken (1 round = bull + bear = 2 turns)

    # --- Research manager ---------------------------------------------------
    investment_plan: str   # written plan in markdown
    decision: dict         # InvestmentDecision as a plain dict (easy to save as JSON)
