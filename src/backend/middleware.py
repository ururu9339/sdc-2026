"""Pure ASGI middleware: request ids and one log line per request; a content-hash ETag for the cacheable
GET listings; and the CORS settings read from the environment.

Logging: request id, method, path (no query string), status and milliseconds. Never request bodies, headers,
CV text or exception messages. Unhandled errors are logged by exception class only and answered with the
standard internal_error body."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import uuid
from typing import Awaitable, Callable

from starlette.datastructures import MutableHeaders

from src.backend.errors import error_response

logger = logging.getLogger("sdc.api")

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
API_PREFIX = "/api/v1"
CACHEABLE_PATHS = frozenset({
    "/api/v1/forecast", "/api/v1/snapshot", "/api/v1/model",
    "/api/v1/taxonomy/skills", "/api/v1/taxonomy/jobs",
    "/forecast", "/snapshot", "/model", "/taxonomy/skills", "/taxonomy/jobs",
})
DEFAULT_CORS = ["*"]


def cors_origins(value: str | None = None) -> list[str]:
    """CORS_ORIGINS: comma-separated origins. Unset or blank means '*'."""
    raw = os.environ.get("CORS_ORIGINS") if value is None else value
    origins = [o.strip().rstrip("/") for o in (raw or "").split(",") if o.strip()]
    return origins or list(DEFAULT_CORS)


def _header(scope: dict, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _etag_matches(if_none_match: str | None, etag: str) -> bool:
    if not if_none_match:
        return False
    for token in (t.strip() for t in if_none_match.split(",")):
        if token == "*" or token == etag or token.removeprefix("W/") == etag:
            return True
    return False


class RequestLogMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]]):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = _header(scope, b"x-request-id")
        rid = incoming if incoming and _VALID_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        started = False
        status_code = 500
        t0 = time.perf_counter()

        async def send_wrapper(message):
            nonlocal started, status_code
            if message["type"] == "http.response.start":
                started = True
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = rid
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:  # answered here, so CORS and the request id still reach the client
            frame = exc.__traceback__
            while frame is not None and frame.tb_next is not None:
                frame = frame.tb_next
            where = f"{frame.tb_frame.f_code.co_filename}:{frame.tb_lineno}" if frame else "unknown"
            logger.error("request_id=%s unhandled=%s at=%s", rid, type(exc).__name__, where)
            if started:
                raise
            status_code = 500
            response = error_response(500, "internal_error", "Internal server error.", None, rid)
            await response(scope, receive, send_wrapper)
        finally:
            logger.info("request_id=%s method=%s path=%s status=%d ms=%.1f", rid, scope.get("method"),
                        scope.get("path"), status_code, (time.perf_counter() - t0) * 1000)


def _is_no_store(scope: dict) -> bool:
    """Report responses (POST /report and GET /report/candidate/{id}) can hold CV-derived data: never stored."""
    path = scope.get("path", "")
    return (scope.get("method") == "POST" and path in ("/report", f"{API_PREFIX}/report")) or \
        path.startswith(("/report/candidate/", f"{API_PREFIX}/report/candidate/"))


class ETagMiddleware:
    """Caching rules. GET listings that change only when the data changes get an ETag (a hash of the body),
    Cache-Control no-cache (revalidate every time), and If-None-Match gets 304 with no body. Report responses
    get Cache-Control no-store, including error responses."""

    def __init__(self, app: Callable[..., Awaitable[None]]):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if _is_no_store(scope):
            async def no_store_send(message):
                if message["type"] == "http.response.start":
                    MutableHeaders(scope=message)["Cache-Control"] = "no-store"
                await send(message)
            return await self.app(scope, receive, no_store_send)
        if scope.get("method") != "GET" or scope.get("path") not in CACHEABLE_PATHS:
            return await self.app(scope, receive, send)
        if_none_match = _header(scope, b"if-none-match")
        held: dict = {}
        chunks: list[bytes] = []

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                held["start"] = message
                return
            if message["type"] != "http.response.body":
                return await send(message)
            chunks.append(message.get("body", b""))
            if message.get("more_body", False):
                return
            start = held.get("start")
            if start is None:  # nothing was started; pass through
                return await send(message)
            body = b"".join(chunks)
            headers = MutableHeaders(scope=start)
            headers["Cache-Control"] = "no-cache"
            if start["status"] == 200:
                etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'
                headers["ETag"] = etag
                if _etag_matches(if_none_match, etag):
                    start["status"] = 304
                    body = b""
            headers["content-length"] = str(len(body))
            await send(start)
            await send({"type": "http.response.body", "body": body, "more_body": False})

        await self.app(scope, receive, send_wrapper)
