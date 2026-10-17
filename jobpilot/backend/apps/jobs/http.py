"""HTTP client for job sources: timeouts, retry with exponential backoff + jitter on
429/5xx/network errors, Retry-After support, and a per-source minimum interval
between requests coordinated through the cache (Redis in production)."""
from __future__ import annotations

import logging
import random
import time

import httpx
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class SourceHTTPError(Exception):
    def __init__(self, message, status_code=None, retryable=False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class SourceHttpClient:
    def __init__(self, throttle_key: str, min_interval: float = 1.0, max_retries: int = 3, backoff_base: float | None = None,
                 timeout: float | None = None, sleep=time.sleep, transport: httpx.BaseTransport | None = None,
                 public_only: bool = False):
        self.throttle_key = f"source-throttle:{throttle_key}"
        self.min_interval = max(0.0, min_interval)
        self.max_retries = max_retries
        self.backoff_base = settings.SOURCE_HTTP_BACKOFF_SECONDS if backoff_base is None else backoff_base
        self.sleep = sleep
        self.client = httpx.Client(
            timeout=timeout or settings.SOURCE_HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": settings.SOURCE_USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
            transport=transport,
            # Re-validated on every request and redirect hop when fetching user-supplied URLs.
            event_hooks={"request": [self._check_public]} if public_only else {},
        )

    @staticmethod
    def _check_public(request):
        from .crawl import UnsafeURL, check_public_url

        try:
            check_public_url(str(request.url))
        except UnsafeURL as exc:
            raise SourceHTTPError(str(exc)) from exc

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _throttle(self):
        if not self.min_interval or not settings.SOURCE_THROTTLE_ENABLED:
            return
        last = cache.get(self.throttle_key)
        now = time.time()
        if last is not None:
            wait = self.min_interval - (now - float(last))
            if wait > 0:
                self.sleep(wait)
        cache.set(self.throttle_key, time.time(), timeout=int(self.min_interval * 10) + 60)

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        attempt = 0
        while True:
            self._throttle()
            try:
                response = self.client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.max_retries:
                    raise SourceHTTPError(f"Network error: {type(exc).__name__}", retryable=True) from exc
                delay = self._delay(attempt)
            else:
                if response.status_code < 400:
                    return response
                retryable = response.status_code in RETRY_STATUSES
                if not retryable or attempt >= self.max_retries:
                    raise SourceHTTPError(
                        f"HTTP {response.status_code} from {response.request.url.host}", status_code=response.status_code,
                        retryable=retryable,
                    )
                delay = self._delay(attempt, response.headers.get("Retry-After"))
            attempt += 1
            logger.info("source_http_retry", extra={"key": self.throttle_key, "attempt": attempt, "delay": round(delay, 2)})
            self.sleep(delay)

    def _delay(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                pass
        return min(self.backoff_base * (2 ** attempt) + random.uniform(0, self.backoff_base), 60.0)

    def get_text(self, url: str, **kwargs) -> str:
        headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", **kwargs.pop("headers", {})}
        response = self.request("GET", url, headers=headers, **kwargs)
        if len(response.content) > 5 * 1024 * 1024:
            raise SourceHTTPError("Page is larger than 5 MB; skipped")
        return response.text

    def get_json(self, url: str, **kwargs):
        response = self.request("GET", url, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise SourceHTTPError("Source returned invalid JSON") from exc
