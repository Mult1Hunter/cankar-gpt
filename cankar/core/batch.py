"""Anthropic Batch API policy - the second transport this repo owns.

`core/http.py`'s `PoliteSession` does this for crawling: one place holds the
user-agent, timeout and rate limit so no caller re-decides them. The Batch API
had grown three call sites (`pairs/cli.py` twice, `evals/cli.py` once) and the
newest one silently lost the named poll constants, the domain-error client
factory, and the submit-time receipt - each of which existed because an earlier
run got burned without it. Stages cannot import each other, so the shared parts
live here.

What is deliberately NOT here: the request builders and the parsers. Those are
per-stage prompt policy (de-styling vs judging) and sharing them would couple two
prompts that must be free to differ.
"""

from __future__ import annotations

import os
from typing import Any

from cankar.core.errors import CankarError

# Exponential backoff between polls. Batches take minutes to hours; starting at 5s
# keeps a small batch responsive and the cap keeps a long one from hammering.
POLL_START_SECONDS = 5
POLL_MAX_SECONDS = 60

# Slovene chars/token, regressed on the real Phase 5 bill rather than taken from
# count_tokens - which was measured against a different model family and read 31%
# low. One definition: the second copy is how two estimates drift apart.
CHARS_PER_TOKEN = 2.09

DEFAULT_MODEL = "claude-sonnet-5"


def client() -> Any:
    """Anthropic client, or a domain error naming the fix.

    `os.environ[...]` raises KeyError, which surfaces as a traceback rather than
    the one-line message that tells an operator to fill in `.env`.
    """
    from anthropic import Anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise CankarError("ANTHROPIC_API_KEY is not set - copy .env.example to .env")
    return Anthropic(api_key=key)


def extract_text(message: dict[str, Any]) -> str:
    """Join every text block of a message.

    NEVER `content[0]["text"]`: with extended thinking enabled the first block is
    a thinking block, so indexing reads the wrong one - and reads it silently,
    because a thinking block is also a plausible-looking dict.
    """
    blocks = message.get("content") or []
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text").strip()
