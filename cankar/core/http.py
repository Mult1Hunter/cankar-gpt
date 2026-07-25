"""Shared polite HTTP transport (ADR 0008).

One place defines the project UA, timeouts, rate limiting, and error raising -
both crawl sources use it; tests inject a fake transport by subclassing.
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

UA = "CankarGPT-corpus-builder/0.1 (+https://nextgen-solutions.xyz; educational project)"
DEFAULT_SLEEP = 0.5  # polite delay between requests, seconds
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 3  # transient-failure retries with exponential backoff
MAX_BACKOFF = 60.0  # cap for server-supplied Retry-After, seconds
RETRYABLE = frozenset({429, 500, 502, 503, 504})


class PoliteSession:
    """requests.Session wrapper: project UA, per-request delay, raise_for_status,
    and retry-with-backoff on transient failures (429/5xx/connection errors,
    honoring Retry-After) - an external review correctly noted the crawlers had
    no retry story."""

    def __init__(
        self,
        sleep: float = DEFAULT_SLEEP,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ):
        self.sleep = sleep
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = UA

    def get(self, url: str, **kwargs: object) -> requests.Response:
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout, **kwargs)  # type: ignore[arg-type]
            except requests.ConnectionError:
                if attempt >= self.retries:
                    raise
                backoff = 2.0**attempt
                logger.warning(f"connection error, retry {attempt + 1} in {backoff}s: {url}")
                time.sleep(backoff)
                continue
            if resp.status_code in RETRYABLE and attempt < self.retries:
                try:  # Retry-After may be an HTTP-date or garbage; cap the sleep
                    backoff = min(float(resp.headers.get("Retry-After", "")), MAX_BACKOFF)
                except ValueError:
                    backoff = 2.0**attempt
                logger.warning(f"HTTP {resp.status_code}, retry {attempt + 1} in {backoff}s: {url}")
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            time.sleep(self.sleep)
            return resp
        raise AssertionError("unreachable")  # loop always returns or raises
