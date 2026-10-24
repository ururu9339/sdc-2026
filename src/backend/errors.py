"""One error shape for every failure: {"error": {"code", "message", "details", "request_id"}}.

Codes are snake_case and tied to the HTTP status: 404 not_found, 422 validation_error, 503 service_unavailable,
500 internal_error. Validation details carry only location, message and type; the submitted values are never
echoed back, so CV text cannot leak through an error. Unhandled errors never expose a traceback."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.backend.models.meta import ErrorEnvelope

CODE_BY_STATUS = {
    400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
    409: "conflict", 413: "payload_too_large", 415: "unsupported_media_type", 422: "validation_error",
    429: "too_many_requests", 500: "internal_error", 503: "service_unavailable",
}
# OpenAPI documentation attached to routes, so clients see the error shape in the schema.
ERROR_RESPONSES: dict = {
    404: {"model": ErrorEnvelope, "description": "Not found (unknown entity, candidate or path)."},
    422: {"model": ErrorEnvelope, "description": "Validation error (bad query, body or path parameter)."},
    503: {"model": ErrorEnvelope, "description": "A dependency is unavailable (for example OCR)."},
}


class ApiError(Exception):
    """Raise to return the standard error body with a chosen status and code."""

    def __init__(self, status_code: int, message: str, details: Any = None, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.details = details
        self.code = code or CODE_BY_STATUS.get(status_code, "error")


def error_body(code: str, message: str, details: Any = None, request_id: str | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details, "request_id": request_id}}


def error_response(status_code: int, code: str, message: str, details: Any = None,
                   request_id: str | None = None) -> JSONResponse:
    return JSONResponse(error_body(code, message, details, request_id), status_code=status_code)


def _request_id(request: Request) -> str | None:
    return (request.scope.get("state") or {}).get("request_id")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return error_response(exc.status_code, exc.code, exc.message, exc.details, _request_id(request))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if isinstance(exc.detail, str):
            message = "Resource not found." if exc.detail == "Not Found" and exc.status_code == 404 else exc.detail
        else:
            message = CODE_BY_STATUS.get(exc.status_code, "error")
        details = None if isinstance(exc.detail, str) else exc.detail
        code = CODE_BY_STATUS.get(exc.status_code, "error")
        return error_response(exc.status_code, code, message, details, _request_id(request))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
                   for e in exc.errors()]
        return error_response(422, "validation_error", "Request validation failed.", details, _request_id(request))
