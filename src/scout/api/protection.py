"""
Protection for the web API, added around the endpoints without changing what they do:

  SecurityHeaders  browser protection headers on every response (no sniffing, no framing of Scout by other
                   sites, HTTPS only, microphone and camera for Scout itself only)
  RateLimiter      how many calls each signed-in shopper may make per action in a minute, for the calls that
                   cost the most (Gemini, image generation, speech). Over the limit the shopper gets a friendly
                   429 and can carry on a few seconds later; other shoppers are not affected.

The counts live in this server's memory. Scout runs a single Cloud Run instance, so they are exact; with more
instances each would count on its own.
"""

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException

TOO_FAST = "You're going a bit fast. Please wait a few seconds and try again."

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy", b"frame-ancestors 'none'"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
    (b"permissions-policy", b"microphone=(self), camera=(self), geolocation=()"),
]


class SecurityHeaders:
    """ASGI middleware: adds the headers as a response starts, so streamed replies are never held back."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                message["headers"] = headers + [h for h in SECURITY_HEADERS if h[0] not in present]
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RateLimiter:
    """At most `limits[action]` calls per shopper and action in any `window_s` seconds (a sliding window)."""

    def __init__(self, limits: dict[str, int], window_s: float = 60.0):
        self._limits = dict(limits)
        self._window = window_s
        self._calls: dict[tuple[str, str], deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def wait_s(self, shopper_id: str, action: str) -> float:
        """Records the call and returns 0 when it is allowed; otherwise how many seconds until it would be."""
        limit = self._limits.get(action)
        if limit is None:
            return 0.0
        now = time.monotonic()
        with self._lock:
            calls = self._calls[(shopper_id, action)]
            while calls and now - calls[0] >= self._window:
                calls.popleft()
            if len(calls) >= limit:
                return self._window - (now - calls[0])
            calls.append(now)
            return 0.0

    def check(self, shopper_id: str, action: str) -> None:
        """Raises a friendly 429 when this shopper has made too many of these calls in the last minute."""
        wait = self.wait_s(shopper_id, action)
        if wait > 0:
            raise HTTPException(status_code=429, detail=TOO_FAST, headers={"Retry-After": str(int(wait) + 1)})
