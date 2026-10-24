"""Shared pieces for the routers: strict query validation and the reusable OpenAPI examples."""
from __future__ import annotations

from typing import Callable

from fastapi import Request

from src.backend.errors import ApiError


def strict_query(allowed: set[str]) -> Callable:
    """Dependency that rejects query parameters the route does not define (422 with the error body), so a typo
    such as 'sortby' is reported instead of silently ignored."""
    def check(request: Request) -> None:
        unknown = sorted(set(request.query_params.keys()) - allowed)
        if unknown:
            raise ApiError(422, "Unknown query parameter(s).", details={"unknown": unknown,
                                                                       "allowed": sorted(allowed)})
    return check
