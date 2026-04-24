"""Shared HTTP retry utility for trace adapters."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    max_retries: int = 5,
    timeout: float = 60,
    client: httpx.Client | None = None,
) -> httpx.Response:
    """HTTP request with exponential backoff + jitter on transient errors.

    Retries on 429/500/502/503/504 and on network-level failures (timeouts,
    connect errors).  Respects ``Retry-After`` on 429s.

    ``max_retries`` counts *retries* after the initial attempt — total
    attempts = max_retries + 1.

    Passing ``client`` reuses its connection pool across calls; the caller
    must set ``follow_redirects=True`` on the client (BTQL uses 303s for
    large payloads).
    """
    resp: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        try:
            if client is not None:
                resp = client.request(
                    method,
                    url,
                    headers=headers,
                    json=json,
                    timeout=timeout,
                )
            else:
                resp = httpx.request(
                    method,
                    url,
                    headers=headers,
                    json=json,
                    timeout=timeout,
                    follow_redirects=True,
                )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            if attempt == max_retries:
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
            if attempt == max_retries:
                break
            wait = _get_retry_wait(resp, attempt)
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
        raise RuntimeError(f"No response after {max_retries + 1} attempts: {url}")
    return resp


def _get_retry_wait(resp: httpx.Response, attempt: int) -> float:
    """Compute wait time, respecting Retry-After header for 429s."""
    if resp.status_code == 429:
        retry_after = resp.headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
    return (2**attempt) + random.uniform(0, 1)
