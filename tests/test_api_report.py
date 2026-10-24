import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from src.backend.db import get_session
from src.backend.main import app
from src.backend.models.candidate import Candidate

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)


def test_post_report_offline_example_is_200_and_labelled(preview_env):
    r = client.post("/report", json={"skills": ["skill:accounting", "skill:excel"]})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "offline_alias" and body["feed"] == "preview" and body["validated"] is False
    assert body["banner"].startswith("PREVIEW:")
    assert body["best_fit_jobs"][0]["job_id"] == "job:accountant"
    assert body["learning_path"]["note"].startswith("The order is a heuristic")


def test_post_report_no_skills_is_200_with_explanation(no_model_env):
    body = client.post("/report", json={}).json()
    assert body["status"] == "no_skills_resolved" and body["best_fit_jobs"] == []
    assert body["feed"] == "none"


def test_post_report_rejects_more_than_100_skills(no_model_env):
    r = client.post("/report", json={"skills": ["skill:python"] * 101})
    assert r.status_code == 422


def test_post_report_rejects_an_oversize_item(no_model_env):
    r = client.post("/report", json={"skills": ["x" * 201]})
    assert r.status_code == 422


def test_post_report_rejects_oversize_text(no_model_env):
    r = client.post("/report", json={"text": "x" * 20001})
    assert r.status_code == 422


def test_post_report_accepts_text_of_exactly_the_limit(no_model_env):
    r = client.post("/report", json={"text": "x" * 20000})
    assert r.status_code == 200


@pytest.mark.parametrize("horizon", [0, 13])
def test_post_report_rejects_horizon_out_of_range(no_model_env, horizon):
    assert client.post("/report", json={"skills": ["skill:python"], "horizon_months": horizon}).status_code == 422


def test_post_report_rejects_unknown_engine(no_model_env):
    assert client.post("/report", json={"skills": ["skill:python"], "engine": "gpt"}).status_code == 422


@pytest.fixture
def candidate_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'candidates.db'}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        cand = Candidate(full_name="Test Person", email="person@example.test",
                         skills="Accounting, Excel;\nQuantum basket weaving",
                         experience="Prepared monthly reports in Excel", summary="Accountant")
        session.add(cand)
        session.commit()
        session.refresh(cand)
        candidate_id = cand.id

    def override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override
    try:
        yield candidate_id
    finally:
        app.dependency_overrides.pop(get_session, None)


def test_candidate_report_uses_skills_and_text(preview_env, candidate_db):
    r = client.get(f"/report/candidate/{candidate_db}")
    assert r.status_code == 200
    body = r.json()
    matched = {x["skill_id"] for x in body["resolved_skills"] if x["status"] == "matched"}
    assert {"skill:accounting", "skill:excel"} <= matched
    assert body["unresolved"] == ["Quantum basket weaving"]
    assert body["warnings"] == []


def test_candidate_report_unknown_id_is_404(preview_env, candidate_db):
    assert client.get(f"/report/candidate/{uuid.uuid4()}").status_code == 404


def test_candidate_report_bad_id_is_422(preview_env, candidate_db):
    assert client.get("/report/candidate/not-a-uuid").status_code == 422


def test_candidate_report_rejects_bad_horizon_and_engine(preview_env, candidate_db):
    assert client.get(f"/report/candidate/{candidate_db}", params={"horizon_months": 0}).status_code == 422
    assert client.get(f"/report/candidate/{candidate_db}", params={"engine": "gpt"}).status_code == 422


def test_taxonomy_skills_lists_the_real_vocabulary():
    body = client.get("/taxonomy/skills").json()
    assert body["count"] == len(body["items"]) >= 100
    assert all(item["id"].startswith("skill:") for item in body["items"])
    python = next(item for item in body["items"] if item["id"] == "skill:python")
    assert python["name"] == "Python" and "python3" in python["alt_labels"]


def test_existing_endpoints_still_respond(no_model_env):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    assert client.get("/api/v1/forecast").json()["status"] == "no_live_model"
