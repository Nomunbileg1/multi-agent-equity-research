"""Researcher team: a bull and a bear debate, then a research manager decides.

    bull -> bear -> bull -> bear -> research manager
    (the loop back to the bull is a conditional edge in src/graph.py)

The bull and bear use the quick model (Haiku) and see all four analyst
reports, the debate so far, and their opponent's latest argument. The research
manager uses the deep model (Sonnet) and returns a structured
`InvestmentDecision`, which includes the written investment plan.
"""

from __future__ import annotations

from typing import Callable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import InvestmentDecision, ResearchState, get_as_of_date

# ---------------------------------------------------------------------------
# Prompts (edit these to change how the debate and the decision work)
# ---------------------------------------------------------------------------

RESEARCHER_SYSTEM_PROMPT = """\
You are the {side_title} Researcher on an equity research team, debating {ticker} \
(as of {as_of_date}) for a medium-term (3-6 month) investment horizon.

Your job: build the strongest evidence-based case {stance} investing in {ticker}.

Rules:
- Use ONLY facts from the four analyst reports. Do not introduce new numbers or events.
- If the {opponent_title} has spoken, start by rebutting their latest argument point by point.
- You may concede points that are clearly correct, but explain why they don't change your thesis.
- Don't repeat your earlier arguments; add new evidence or sharpen existing points.
- Be direct and specific. Under 250 words. Plain markdown, no title.
"""

MANAGER_SYSTEM_PROMPT = """\
You are the Research Manager. You judge a bull-versus-bear debate about {ticker} \
(as of {as_of_date}) and make the final call for a medium-term (3-6 month) horizon.

How to judge:
- Decide on the QUALITY OF EVIDENCE, not on who argued last, longest or loudest.
- Check each side's claims against the analyst reports; discount anything unsupported.
- Commit to BUY, HOLD or SELL. Choose HOLD only when the evidence is genuinely balanced, \
never as a way to avoid deciding.

Fill in every field of the structured decision. The `investment_plan` field is the \
written report, in markdown, with these sections:
## Recommendation
(the rating, confidence and time horizon, plus a one-paragraph thesis)
## Rationale
## Strongest Counter-Argument
## Key Risks
## What Would Change My Mind
Keep the plan under 600 words and cite specific figures from the reports.
"""


def format_analyst_reports(state: ResearchState) -> str:
    """Combine the four analyst reports into one block for the researchers' prompts."""
    sections = [
        ("Market / Technical Analyst", "market_report"),
        ("Fundamentals Analyst", "fundamentals_report"),
        ("News Analyst", "news_report"),
        ("Sentiment Analyst", "sentiment_report"),
    ]
    return "\n\n".join(
        f"=== {title} ===\n{state.get(key) or 'Report unavailable.'}" for title, key in sections
    )


# ---------------------------------------------------------------------------
# Bull and bear
# ---------------------------------------------------------------------------

def make_researcher(
    side: Literal["bull", "bear"],
    opponent: Literal["bull", "bear"],
    llm: BaseChatModel | None = None,
) -> Callable[[ResearchState], dict]:
    """Create the bull or bear researcher node.

    Each turn it appends its argument to `debate_history`, stores it in
    `{side}_last`, and increments `debate_count` (which the graph uses to
    decide when the debate is over).
    """
    if llm is None:
        from src.llm import quick_llm  # imported here so tests never need an API key
        llm = quick_llm

    side_title, opponent_title = side.capitalize(), opponent.capitalize()

    def researcher_node(state: ResearchState) -> dict:
        history = state.get("debate_history", "")
        turn = state.get("debate_count", 0) + 1
        opponent_last = state.get(f"{opponent}_last", "")

        system = SystemMessage(
            RESEARCHER_SYSTEM_PROMPT.format(
                side_title=side_title,
                opponent_title=opponent_title,
                ticker=state["ticker"],
                as_of_date=get_as_of_date(state),
                stance="for" if side == "bull" else "against",
            )
        )
        request = HumanMessage(
            f"ANALYST REPORTS\n\n{format_analyst_reports(state)}\n\n"
            f"DEBATE SO FAR\n\n{history or '(The debate has not started. You open.)'}\n\n"
            f"{opponent_title.upper()}'S LATEST ARGUMENT\n\n{opponent_last or '(none yet)'}\n\n"
            f"Make your {'opening' if not history else 'next'} argument as the {side_title}."
        )
        try:
            argument = llm.invoke([system, request]).text.strip()
        except Exception as error:  # noqa: BLE001  (keep the debate moving even if one turn fails)
            argument = f"(The {side} researcher could not respond: {type(error).__name__}: {error})"

        entry = f"### {side_title} (turn {turn})\n\n{argument}"
        return {
            "debate_history": f"{history}\n\n{entry}".strip(),
            f"{side}_last": argument,
            "debate_count": turn,
        }

    researcher_node.__name__ = f"{side}_researcher"
    return researcher_node


# ---------------------------------------------------------------------------
# Research manager
# ---------------------------------------------------------------------------

def make_research_manager(llm: BaseChatModel | None = None) -> Callable[[ResearchState], dict]:
    """Create the research manager node (deep model + structured output).

    `with_structured_output(..., method="json_schema")` makes Claude return JSON
    that matches `InvestmentDecision`, which LangChain parses into a Pydantic
    object. One call returns both the rating and the written plan.
    """
    if llm is None:
        from src.llm import deep_llm
        llm = deep_llm
    structured_llm = llm.with_structured_output(InvestmentDecision, method="json_schema")

    def research_manager(state: ResearchState) -> dict:
        system = SystemMessage(
            MANAGER_SYSTEM_PROMPT.format(ticker=state["ticker"], as_of_date=get_as_of_date(state))
        )
        request = HumanMessage(
            f"ANALYST REPORTS\n\n{format_analyst_reports(state)}\n\n"
            f"FULL DEBATE TRANSCRIPT\n\n{state.get('debate_history') or '(no debate took place)'}\n\n"
            "Judge the debate and give your decision."
        )
        try:
            decision: InvestmentDecision = structured_llm.invoke([system, request])
            return {"investment_plan": decision.investment_plan, "decision": decision.model_dump()}
        except Exception as error:  # noqa: BLE001  (report the failure instead of crashing)
            message = f"The research manager could not reach a decision ({type(error).__name__}: {error})."
            return {
                "investment_plan": f"## Recommendation\n\nUnavailable. {message}",
                "decision": {"rating": "UNAVAILABLE", "error": message},
            }

    return research_manager
