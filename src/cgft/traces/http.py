"""Shared HTTP retry utility for trace adapters."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = (429, 502, 503, 504)


def request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    max_retries: int = 5,
    timeout: float = 60,
) -> httpx.Response:
    """HTTP request with exponential backoff + jitter on transient errors.

    Retries on 429/502/503/504 status codes and on network-level failures
    (timeouts, connection errors).  Jitter prevents thundering herd when
    multiple workers hit the same API concurrently.
    """
    resp: httpx.Response | None = None
    for attempt in range(max_retries):
        try:
            resp = httpx.request(
                method,
                url,
                headers=headers,
                json=json,
                timeout=timeout,
                follow_redirects=True,
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            if attempt == max_retries - 1:
                raise
            wait = (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "Request failed (%s), retry %d/%d in %.1fs: %s",
                exc.__class__.__name__,
                attempt + 1,
                max_retries,
                wait,
                url,
            )
            time.sleep(wait)
            continue
        if resp.status_code in _RETRYABLE_STATUS:
            wait = (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "HTTP %d, retry %d/%d in %.1fs: %s",
                resp.status_code,
                attempt + 1,
                max_retries,
                wait,
                url,
            )
            time.sleep(wait)
            continue
        break
    if resp is None:
        raise RuntimeError(f"No response after {max_retries} retries: {url}")
    return resp
