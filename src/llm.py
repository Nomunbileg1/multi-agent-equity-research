"""Claude client setup.

Two models, routed by job:
    quick_llm: Haiku, for the analysts and the bull/bear researchers
    deep_llm:  Sonnet, for the research manager's final judgment

Keys are read from `.env` (never hard-coded). If LangSmith variables are set
there too, every call is traced automatically; no extra code is needed.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src import config

# Load ANTHROPIC_API_KEY (and optional LANGSMITH_*) from .env into the environment.
load_dotenv()

quick_llm = ChatAnthropic(
    model=config.QUICK_MODEL,
    max_tokens=config.QUICK_MAX_TOKENS,
    output_config={"effort": config.QUICK_EFFORT},
)

deep_llm = ChatAnthropic(
    model=config.DEEP_MODEL,
    max_tokens=config.DEEP_MAX_TOKENS,
    output_config={"effort": config.DEEP_EFFORT},
)
