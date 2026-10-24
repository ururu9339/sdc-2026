import pytest
from fastapi.testclient import TestClient

from src.backend import main
from src.backend.api import candidates


@pytest.fixture
def site(tmp_path, monkeypatch, no_model_env):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Currvix</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    monkeypatch.setattr(main, "FRONTEND_DIST", dist)
    return TestClient(main.create_app())


def test_root_and_client_routes_serve_the_app(site):
    for path in ("/", "/explore", "/explore/deep/link"):
        r = site.get(path)
        assert r.status_code == 200 and "<title>Currvix</title>" in r.text, path


def test_static_assets_are_served_and_missing_files_are_404(site):
    assert site.get("/assets/app.js").text == "console.log(1)"
    assert site.get("/assets/missing.js").status_code == 404


def test_api_paths_never_fall_back_to_the_app(site):
    r = site.get("/api/v1/not-a-route")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert site.get("/api/v1/health").json()["status"] == "ok"


def test_security_headers_on_pages_and_api(site):
    for path in ("/", "/api/v1/health"):
        h = site.get(path).headers
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["X-Frame-Options"] == "DENY"
        assert h["Content-Security-Policy"] == "frame-ancestors 'none'"
        assert h["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_no_site_mounted_without_a_build(tmp_path, monkeypatch, no_model_env):
    monkeypatch.setattr(main, "FRONTEND_DIST", tmp_path / "missing")
    assert TestClient(main.create_app()).get("/explore").status_code == 404


def test_oversized_upload_is_rejected(no_model_env, monkeypatch):
    monkeypatch.setattr(candidates, "MAX_UPLOAD_BYTES", 10)
    r = TestClient(main.create_app()).post("/api/v1/candidates/pdf",
                                           files={"file": ("cv.pdf", b"%PDF-" + b"x" * 20, "application/pdf")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
