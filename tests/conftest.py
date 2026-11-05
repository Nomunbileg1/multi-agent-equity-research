"""Shared pytest setup: keep the tests offline, free and quiet.

This runs before any test module imports `src`, so:
- LangSmith tracing is switched off (fake runs shouldn't be uploaded), and
- no real API key is needed; the fake models never call Anthropic.

`load_dotenv()` never overrides variables that are already set, so these
values win over anything in a local .env file.
"""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ.setdefault("ANTHROPIC_API_KEY", "")
