"""API contract: versioned paths and aliases, CORS, error body, request ids, health, pagination, sort, search,
ETag and the preview banner. Fixtures are hand-written (conftest.py) and live in tmp_path."""
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.main import app, create_app
from src.backend.middleware import cors_origins
from src.training import config as C
from tests.conftest import PREVIEW_REASON

client = TestClient(app)
BANNER_PREFIX = "PREVIEW: unvalidated candidate forecasts"


# ---- versioned paths and aliases -------------------------------------------------------------------
def test_every_public_route_is_documented_under_api_v1_only():
    paths = app.openapi()["paths"]
    assert "/api/v1/forecast" in paths and "/api/v1/forecast/{entity_id}" in paths
    assert "/api/v1/health" in paths and "/api/v1/taxonomy/jobs" in paths and "/api/v1/report" in paths
    assert "/forecast" not in paths and "/health" not in paths and "/report" not in paths
    assert all(p.startswith("/api/v1/") for p in paths)


def test_unprefixed_alias_and_prefixed_path_serve_the_same_forecast(preview_env):
    a = client.get("/forecast", params={"limit": 2}).json()
    b = client.get("/api/v1/forecast", params={"limit": 2}).json()
    assert a == b and b["validated"] is False


@pytest.mark.parametrize("path", ["/api/v1/forecast/job:accountant", "/api/v1/explain/job:accountant",
                                  "/api/v1/history/job:accountant", "/api/v1/model", "/api/v1/sources",
                                  "/api/v1/snapshot", "/api/v1/taxonomy/skills", "/api/v1/taxonomy/jobs"])
def test_v1_routes_respond(preview_env, path):
    assert client.get(path).status_code == 200


def test_operation_ids_are_stable_and_unique():
    ops = [op["operationId"] for p in app.openapi()["paths"].values() for op in p.values()]
    assert len(ops) == len(set(ops))
    assert "list_forecasts" in ops and "create_report" in ops and "get_health" in ops


def test_post_report_example_is_in_openapi():
    body = app.openapi()["paths"]["/api/v1/report"]["post"]["requestBody"]["content"]["application/json"]
    assert body.get("examples") or body.get("example")


# ---- CORS -------------------------------------------------------------------------------------------
def _preflight(client_, origin):
    return client_.options("/api/v1/forecast", headers={"Origin": origin,
                                                        "Access-Control-Request-Method": "GET"})


def test_cors_default_is_wildcard_without_credentials(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    r = _preflight(TestClient(create_app()), "https://frontend.example")
    assert r.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in r.headers


def test_cors_explicit_origins_allow_credentials_and_expose_headers(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example.test, https://b.example.test/")
    c = TestClient(create_app())
    r = _preflight(c, "https://a.example.test")
    assert r.headers.get("access-control-allow-origin") == "https://a.example.test"
    assert r.headers.get("access-control-allow-credentials") == "true"
    assert _preflight(c, "https://evil.example").headers.get("access-control-allow-origin") is None
    r = c.get("/api/v1/health", headers={"Origin": "https://b.example.test"})
    assert "X-Request-ID" in r.headers.get("access-control-expose-headers", "")
    assert "ETag" in r.headers.get("access-control-expose-headers", "")


def test_cors_origins_parsing():
    assert cors_origins("  ") == ["*"]
    assert cors_origins("https://x.test/,https://y.test") == ["https://x.test", "https://y.test"]


# ---- error body ----------------------------------------------------------------------------------------
def _assert_error(body, code):
    assert set(body) == {"error"}
    err = body["error"]
    assert err["code"] == code and isinstance(err["message"], str)
    assert "details" in err and "request_id" in err


def test_unknown_entity_is_404_with_error_body(no_model_env):
    r = client.get("/api/v1/forecast/not_a_real_entity")
    assert r.status_code == 404
    _assert_error(r.json(), "not_found")


def test_unknown_route_is_404_with_error_body():
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404
    _assert_error(r.json(), "not_found")


def test_validation_error_is_422_and_does_not_echo_the_input(no_model_env):
    secret = "CV-SECRET-TEXT-12345"
    r = client.post("/api/v1/report", json={"skills": ["skill:python"], "horizon_months": 0, "text": secret})
    assert r.status_code == 422
    body = r.json()
    _assert_error(body, "validation_error")
    assert isinstance(body["error"]["details"], list) and body["error"]["details"][0]["loc"]
    assert secret not in r.text


def test_unknown_query_parameter_is_422(preview_env):
    r = client.get("/api/v1/forecast", params={"sortby": "name"})
    assert r.status_code == 422
    _assert_error(r.json(), "validation_error")
    assert r.json()["error"]["details"]["unknown"] == ["sortby"]


def test_unhandled_error_is_500_without_traceback_or_message():
    boom = FastAPI()
    secret = "SECRET-CV-TEXT-DO-NOT-LEAK"
    from src.backend.errors import install_error_handlers
    from src.backend.middleware import ETagMiddleware, RequestLogMiddleware

    async def route():
        raise RuntimeError(secret)

    boom.add_api_route("/api/v1/boom", route, methods=["GET"])
    boom.add_middleware(ETagMiddleware)
    boom.add_middleware(RequestLogMiddleware)
    install_error_handlers(boom)
    r = TestClient(boom, raise_server_exceptions=False).get("/api/v1/boom")
    assert r.status_code == 500
    _assert_error(r.json(), "internal_error")
    assert r.json()["error"]["message"] == "Internal server error."
    assert secret not in r.text and "Traceback" not in r.text
    assert r.headers.get("x-request-id")


# ---- request ids and logging -----------------------------------------------------------------------
def test_incoming_request_id_is_honoured_and_invalid_one_is_replaced():
    r = client.get("/api/v1/health", headers={"X-Request-ID": "front-end-42"})
    assert r.headers["X-Request-ID"] == "front-end-42"
    r = client.get("/api/v1/health", headers={"X-Request-ID": "bad id with spaces"})
    assert r.headers["X-Request-ID"] != "bad id with spaces" and len(r.headers["X-Request-ID"]) == 32
    assert client.get("/api/v1/health", headers={"X-Request-ID": "x" * 129}).headers["X-Request-ID"] != "x" * 129


def test_request_log_line_has_id_method_path_status_and_no_query(caplog):
    caplog.set_level(logging.INFO, logger="sdc.api")
    client.get("/api/v1/health", headers={"X-Request-ID": "log-test-1"}, params={"q": "secret-query-value"})
    lines = [r.getMessage() for r in caplog.records if r.name == "sdc.api" and "log-test-1" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    for part in ("method=GET", "path=/api/v1/health", "status=200", "ms="):
        assert part in line
    assert "secret-query-value" not in line


# ---- health ------------------------------------------------------------------------------------------
@pytest.fixture
def no_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SERVING_DIR", tmp_path / "no-serving")
    monkeypatch.setattr(C, "DATA_LAYOUT", "env")


def test_health_with_no_model_is_ok_and_explains(no_model_env, no_bundle):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["ready"] is True and body["api_version"] == "v1"
    d = body["data"]
    assert d["live_model"] is None and d["feed"] == "none" and d["n_forecasts"] == 0
    assert d["last_gate"] == {"passed": False, "reason": PREVIEW_REASON}
    assert d["gate_profile"] == "demo" and d["data_layout"] == "env" and d["bundle"] is None
    assert d["matching_engine_available"]["offline_alias"] is True
    assert d["matching_engine_available"]["qwen_pgvector"] is False  # no DATABASE_URL in tests
    assert set(d["sources_with_data"]) == {"job_postings", "market_signal"}


def test_health_with_live_model(live_model_env, no_bundle):
    d = client.get("/api/v1/health").json()["data"]
    assert d["feed"] == "live" and d["live_model"] == "v1" and d["last_gate"]["passed"] is True
    assert d["forecast_as_of"] == "2026-09-14" and d["n_forecasts"] == 3


def test_health_preview_feed(tmp_path, monkeypatch, no_bundle):
    from tests.conftest import build_env
    build_env(tmp_path, monkeypatch, live=False, preview=True)
    d = client.get("/api/v1/health").json()["data"]
    assert d["feed"] == "preview" and d["live_model"] is None and d["n_forecasts"] == 3
    assert d["forecast_as_of"] == "2026-09-14"


def test_health_is_200_and_not_ready_when_the_store_is_unreadable(no_model_env, no_bundle, monkeypatch):
    from src.backend.services import forecast_store as store

    def broken():
        raise OSError("boom")
    monkeypatch.setattr(store, "health_facts", broken)
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["ready"] is False
    assert "boom" not in r.text


def test_health_reports_bundle_summary_when_manifest_exists(no_model_env, tmp_path, monkeypatch):
    import json
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "manifest.json").write_text(json.dumps({"generated_at": "2026-10-25T00:00:00+00:00", "as_of": "2026-09-14",
                                                       "feed": "preview", "validated": False, "files": [{}, {}],
                                                       "total_bytes": 1234, "note": "snapshot for serving"}))
    monkeypatch.setattr(C, "SERVING_DIR", serving)
    b = client.get("/api/v1/health").json()["data"]["bundle"]
    assert b == {"generated_at": "2026-10-25T00:00:00+00:00", "as_of": "2026-09-14", "feed": "preview",
                 "model_version": None, "validated": False, "n_files": 2, "total_bytes": 1234,
                 "note": "snapshot for serving"}


def test_health_alias_matches_v1(no_model_env, no_bundle):
    assert client.get("/health").json() == client.get("/api/v1/health").json()


# ---- pagination, sort, search ------------------------------------------------------------------------
def test_pagination_envelope(preview_env):
    body = client.get("/api/v1/forecast", params={"limit": 2, "offset": 0}).json()
    assert body["total"] == 3 and body["count"] == 2 and body["offset"] == 0 and body["limit"] == 2
    page2 = client.get("/api/v1/forecast", params={"limit": 2, "offset": 2}).json()
    assert page2["count"] == 1 and page2["offset"] == 2
    ids = [f["entity_id"] for f in body["forecasts"]] + [f["entity_id"] for f in page2["forecasts"]]
    assert len(ids) == len(set(ids)) == 3


def test_limit_above_500_is_422(preview_env):
    assert client.get("/api/v1/forecast", params={"limit": 501}).status_code == 422
    assert client.get("/api/v1/forecast", params={"limit": 500}).status_code == 200


def test_default_sort_is_largest_growth_first_and_no_number_last(preview_env):
    ents = [f["entity_id"] for f in client.get("/api/v1/forecast").json()["forecasts"]]
    assert ents == ["job:ai_engineer", "job:accountant", "job:business_analyst"]  # p50 0.25, 0.02, none


def test_sort_growth_ascending_and_names(preview_env):
    asc = [f["entity_id"] for f in client.get("/api/v1/forecast", params={"sort": "growth"}).json()["forecasts"]]
    assert asc == ["job:accountant", "job:ai_engineer", "job:business_analyst"]
    names = [f["entity_id"] for f in client.get("/api/v1/forecast", params={"sort": "name"}).json()["forecasts"]]
    assert names == ["job:accountant", "job:ai_engineer", "job:business_analyst"]
    rev = [f["entity_id"] for f in client.get("/api/v1/forecast", params={"sort": "-name"}).json()["forecasts"]]
    assert rev == ["job:business_analyst", "job:ai_engineer", "job:accountant"]
    assert client.get("/api/v1/forecast", params={"sort": "random"}).status_code == 422


def test_search_matches_id_and_taxonomy_name_case_insensitively(preview_env):
    by_id = client.get("/api/v1/forecast", params={"q": "AI_ENG"}).json()
    assert [f["entity_id"] for f in by_id["forecasts"]] == ["job:ai_engineer"] and by_id["total"] == 1
    by_name = client.get("/api/v1/forecast", params={"q": "accountant"}).json()
    assert [f["entity_id"] for f in by_name["forecasts"]] == ["job:accountant"]


def test_min_confidence_keeps_at_or_above(live_model_env):
    body = client.get("/api/v1/forecast", params={"min_confidence": "high"}).json()
    assert [f["entity_id"] for f in body["forecasts"]] == ["job:ai_engineer"]  # newest run only
    low = client.get("/api/v1/forecast", params={"min_confidence": "low"}).json()
    assert low["total"] == 2  # business_analyst has no confidence and is excluded
    assert client.get("/api/v1/forecast", params={"min_confidence": "maybe"}).status_code == 422


def test_taxonomy_jobs_search_and_paging():
    body = client.get("/api/v1/taxonomy/jobs", params={"q": "analyst", "limit": 5}).json()
    assert body["kind"] == "job" and body["total"] >= 1
    assert all("analyst" in (i["name"] + " ".join(i["alt_labels"])).lower() for i in body["items"])
    all_jobs = client.get("/api/v1/taxonomy/jobs", params={"limit": 500}).json()
    assert all_jobs["total"] == 56 and all_jobs["count"] == 56  # the taxonomy has 56 job entries
    page = client.get("/api/v1/taxonomy/jobs", params={"limit": 2, "offset": 2}).json()
    assert page["count"] == 2 and page["offset"] == 2 and page["total"] == 56
    assert client.get("/api/v1/taxonomy/jobs", params={"limit": 0}).status_code == 422


def test_taxonomy_skills_is_paged_and_defaults_to_all():
    body = client.get("/api/v1/taxonomy/skills").json()
    assert body["kind"] == "skill" and body["total"] == body["count"] == len(body["items"]) >= 100


# ---- ETag and caching -----------------------------------------------------------------------------
def test_etag_and_304_on_forecast_listing(preview_env):
    first = client.get("/api/v1/forecast")
    etag = first.headers["ETag"]
    assert first.headers["Cache-Control"] == "no-cache" and etag.startswith('"') and etag.endswith('"')
    again = client.get("/api/v1/forecast", headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b"" and again.headers["ETag"] == etag
    weak = client.get("/api/v1/forecast", headers={"If-None-Match": f"W/{etag}"})
    assert weak.status_code == 304
    other = client.get("/api/v1/forecast", params={"limit": 1}, headers={"If-None-Match": etag})
    assert other.status_code == 200 and other.headers["ETag"] != etag  # a different query is a different body


@pytest.mark.parametrize("path", ["/api/v1/snapshot", "/api/v1/model", "/api/v1/taxonomy/skills",
                                  "/api/v1/taxonomy/jobs"])
def test_listing_endpoints_revalidate(preview_env, path):
    r = client.get(path)
    assert r.headers["Cache-Control"] == "no-cache" and "ETag" in r.headers
    assert client.get(path, headers={"If-None-Match": r.headers["ETag"]}).status_code == 304


def test_report_is_never_stored(preview_env):
    r = client.post("/api/v1/report", json={"skills": ["skill:accounting"]})
    assert r.status_code == 200 and r.headers["Cache-Control"] == "no-store" and "ETag" not in r.headers


# ---- the preview banner is on every forecast-bearing response ------------------------------------------
def test_every_forecast_bearing_response_keeps_the_preview_banner(preview_env):
    responses = [
        client.get("/api/v1/forecast").json(),
        client.get("/api/v1/forecast/job:ai_engineer").json(),
        client.get("/api/v1/explain/job:ai_engineer").json(),
        client.post("/api/v1/report", json={"skills": ["skill:accounting", "skill:excel"]}).json(),
    ]
    for body in responses:
        assert body["feed"] == "preview" and body["validated"] is False
        assert body["banner"] and body["banner"].startswith(BANNER_PREFIX)
