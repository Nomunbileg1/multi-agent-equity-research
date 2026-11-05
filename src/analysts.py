"""Analyst team: four tool-calling agents that each write a short report.

Each analyst is a small agent with its own private loop:

    START -> llm --(asked for tools?)--> tools
              ^                            |
              +----------------------------+
              |
        (no tool calls)
              v
             END   -> the last message is the report

LangGraph concepts used here:
- MessagesState: a ready-made state with one `messages` list. Its reducer
  (`add_messages`) *appends* new messages instead of replacing the list, so
  the conversation builds up turn by turn.
- ToolNode: a prebuilt node that runs whatever tool calls the LLM asked for
  and adds the results to `messages` as ToolMessages.
- tools_condition: a prebuilt router. If the last LLM message contains tool
  calls it goes to "tools"; otherwise it goes to END.

The outer research graph (src/graph.py) treats each analyst as one ordinary
node, so the four loops run in parallel without seeing each other's messages.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src import config
from src.state import ResearchState, get_as_of_date
from src.tools import fetch_fundamentals, fetch_market_data, fetch_news, fetch_sentiment_data

# ---------------------------------------------------------------------------
# Prompts (edit these to change how the analysts think and write)
# ---------------------------------------------------------------------------

ANALYST_SYSTEM_PROMPT = """\
You are the {role} on an equity research team analyzing {ticker} as of {as_of_date}, \
for a medium-term (3-6 month) investment horizon.

Your focus: {focus}

How to work:
- Start by calling your tools to gather data. You have at most {max_rounds} rounds \
of tool calls; fetch what you need efficiently. Never repeat a tool call with the \
same arguments: the data will not change.
- Use ONLY data returned by your tools. Never invent numbers, events, quotes or sources. \
If something is missing, say it is unavailable.
- If a tool returns "DATA UNAVAILABLE", say so plainly, work with whatever else you \
have, and give a Neutral verdict if you have nothing usable.
- Cite specific figures from the data.

Report format (under {max_words} words in total, including the table; markdown):
1. A 2-3 sentence summary.
2. Key findings as short bullet points with numbers.
3. A table: | Factor | Observation | Signal |  (Signal is Bullish, Bearish or Neutral)
4. A final line in exactly this form:
   Verdict: Bullish|Bearish|Neutral - <one-sentence reason>
"""

REPORT_MAX_WORDS = 400

# Focus areas, tools and output keys for each analyst.
MARKET_FOCUS = (
    "price trend versus the 50- and 200-day moving averages; momentum (RSI, MACD); "
    "recent price action and volume spikes; position in the 52-week range."
)
FUNDAMENTALS_FOCUS = (
    "valuation versus growth (P/E, forward P/E, PEG); revenue and margin trend over the "
    "last 4 quarters; balance-sheet strength and free cash flow; earnings quality."
)
NEWS_FOCUS = (
    "material company-specific events (earnings, guidance, products, regulation, "
    "management, M&A). IGNORE headlines that are not about this company. Weigh the "
    "likely impact over the next few months, and check how the stock price has reacted."
)
SENTIMENT_FOCUS = (
    "Wall Street consensus and its trend; upside to analyst price targets; short "
    "interest; tone of recent headlines about the company (ignore unrelated ones); "
    "crowded versus contrarian signals."
)


# ---------------------------------------------------------------------------
# The agent loop (one small graph per analyst)
# ---------------------------------------------------------------------------

def _count_tool_rounds(messages: list) -> int:
    """How many times the LLM has asked for tools so far in this conversation."""
    return sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)


def build_analyst_agent(llm: BaseChatModel, tools: list[BaseTool], max_rounds: int):
    """Compile the small llm <-> tools loop for one analyst.

    After `max_rounds` rounds of tool calls, the LLM is called with
    `tool_choice="none"`, which forces it to stop researching and write the
    report. This caps the cost of each analyst.
    """
    llm_with_tools = llm.bind_tools(tools)
    llm_must_answer = llm.bind_tools(tools, tool_choice={"type": "none"})

    def call_llm(state: MessagesState) -> dict:
        model = llm_must_answer if _count_tool_rounds(state["messages"]) >= max_rounds else llm_with_tools
        response = model.invoke(state["messages"])
        return {"messages": [response]}  # appended to the list by add_messages

    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)  # -> "tools" or END
    builder.add_edge("tools", "llm")  # after running tools, let the LLM read the results
    return builder.compile()


# ---------------------------------------------------------------------------
# Factory: turn an agent into a node of the outer research graph
# ---------------------------------------------------------------------------

def make_analyst(
    role: str,
    tools: list[BaseTool],
    output_key: str,
    focus: str,
    llm: BaseChatModel | None = None,
    max_rounds: int = config.MAX_TOOL_ROUNDS,
) -> Callable[[ResearchState], dict]:
    """Create an analyst node for the research graph.

    The returned function reads `ticker` and `as_of_date` from the shared state,
    runs this analyst's private agent loop, and returns only `{output_key: report}`.

    `llm` defaults to the quick (Haiku) model; tests pass in a fake model instead.
    """
    if llm is None:
        from src.llm import quick_llm  # imported here so tests never need an API key
        llm = quick_llm
    agent = build_analyst_agent(llm, tools, max_rounds)

    def analyst_node(state: ResearchState) -> dict:
        ticker = state["ticker"]
        system = SystemMessage(
            ANALYST_SYSTEM_PROMPT.format(
                role=role,
                ticker=ticker,
                as_of_date=get_as_of_date(state),
                focus=focus,
                max_rounds=max_rounds,
                max_words=REPORT_MAX_WORDS,
            )
        )
        request = HumanMessage(f"Research {ticker} with your tools, then write your report.")
        try:
            result = agent.invoke({"messages": [system, request]})
            report = result["messages"][-1].text.strip()
            if not report:
                raise ValueError("the model returned an empty report")
        except Exception as error:  # noqa: BLE001  (one failed analyst must not crash the run)
            report = (
                f"**{role}: data unavailable.** The analysis could not be completed "
                f"({type(error).__name__}: {error}).\n\n"
                "Verdict: Neutral - no usable data."
            )
        return {output_key: report}

    analyst_node.__name__ = output_key.replace("_report", "_analyst")
    return analyst_node


def make_analyst_team(llm: BaseChatModel | None = None) -> dict[str, Callable[[ResearchState], dict]]:
    """Build all four analyst nodes, keyed by their node name in the graph."""
    return {
        "market_analyst": make_analyst(
            "Market / Technical Analyst", [fetch_market_data], "market_report", MARKET_FOCUS, llm
        ),
        "fundamentals_analyst": make_analyst(
            "Fundamentals Analyst", [fetch_fundamentals, fetch_market_data],
            "fundamentals_report", FUNDAMENTALS_FOCUS, llm,
        ),
        "news_analyst": make_analyst(
            "News Analyst", [fetch_news, fetch_market_data], "news_report", NEWS_FOCUS, llm
        ),
        "sentiment_analyst": make_analyst(
            "Sentiment Analyst", [fetch_sentiment_data, fetch_news],
            "sentiment_report", SENTIMENT_FOCUS, llm,
        ),
    }
