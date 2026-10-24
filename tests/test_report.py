"""Offline report scoring against hand-computed numbers, feed/banner handling, unresolved inputs, empty cases and
the constants that must stay equal to the embeddings team's values. Fixtures live in tmp_path (conftest.py);
the job-skill graph there is a 2-job, 3-skill fixture."""
import math
import re
from pathlib import Path

import pytest

from src.backend.services import forecast_store as store
from src.backend.services import report as service
from src.pipeline.store import AppendOnlyLog
from tests.conftest import PREVIEW_REASON, _record

ROOT = Path(__file__).resolve().parents[1]

# Hand-computed from tests/conftest.py GRAPH (job_skill_graph.csv in the fixture):
#   job:accountant  n=10 postings: accounting 10/10, excel 7/10, python 4/10
#   job:ai_engineer n=9 postings:  python 9/9, accounting 1/9
# Weight = min(n/20, 1) * share. Accountant: accounting 0.5, excel 0.35, python 0.2 (total 1.05).
# AI engineer: python 0.45, accounting 0.05 (total 0.5).


def _s(r, key):
    return [x[key] for x in r]


def _preview_skill_record(p50: float, entity: str = "skill:python"):
    return _record(model_version="preview:v1", feed="preview", validated=False, preview_reason=PREVIEW_REASON,
                   gate_profile="demo", forecast_id=f"preview:v1:{entity}:2026-09-14", as_of="2026-09-14",
                   entity_id=entity, kind="skill", status="ok", reason=None, p50=p50, p10=p50 - 0.05,
                   p90=p50 + 0.05, direction="emerging", confidence="low", signal_quality="noise_like",
                   dispersion_ratio=1.0, source_contributions={"job_postings": 0.0})


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)


# ---- hand-computed example, preview feed --------------------------------------------------------
def test_single_reachable_job_with_preview_growth(preview_env):
    r = service.build_report(["skill:accounting", "skill:excel"])
    assert r["engine"] == "offline_alias" and r["feed"] == "preview" and r["validated"] is False
    assert r["banner"].startswith("PREVIEW:") and r["status"] == "ok"
    best = r["best_fit_jobs"][0]
    assert best["job_id"] == "job:accountant" and best["name"] == "Accountant"
    assert best["coverage"] == pytest.approx(0.85 / 1.05, abs=1e-4)  # 0.8095
    assert _s(best["matched"], "skill_id") == ["skill:accounting", "skill:excel"]
    assert _s(best["missing"], "skill_id") == ["skill:python"]
    assert best["growth"] == pytest.approx(math.exp(0.02) - 1)
    assert best["forecast_status"] == "ok" and best["direction"] == "stable"
    assert r["targets"] == ["job:accountant"]
    # Only the ai_engineer job is left out: its coverage is 0.1, so it is not in reach and not a target.
    assert [j["job_id"] for j in r["best_fit_jobs"]] == ["job:accountant"]


def test_gap_score_without_skill_growth_is_weight(preview_env):
    r = service.build_report(["skill:accounting", "skill:excel"])
    assert len(r["skill_gaps"]) == 1
    gap = r["skill_gaps"][0]
    assert gap["skill_id"] == "skill:python"
    assert gap["score"] == pytest.approx(0.2)            # 0.2 * (1 + 0): no forecast for python
    assert gap["weight"] == pytest.approx(0.2)
    assert gap["share"] == pytest.approx(0.4)            # 4 of 10 accountant postings
    assert gap["forecast_status"] == "no_forecast" and gap["growth"] is None
    assert gap["for_jobs"] == ["job:accountant"] and gap["stage"] == "Start here"
    assert gap["why"] == ("Appears in 40% of Accountant postings in our data; no forecast available")


def test_gap_score_with_skill_growth_and_why_text(preview_env):
    AppendOnlyLog(preview_env["preview_path"]).append_many([_preview_skill_record(0.1)])
    r = service.build_report(["skill:accounting", "skill:excel"])
    gap = r["skill_gaps"][0]
    assert gap["score"] == pytest.approx(0.2 * math.exp(0.1), abs=1e-4)  # 0.2 * (1 + (e^0.1 - 1))
    assert gap["growth"] == pytest.approx(math.exp(0.1) - 1)
    assert gap["direction"] == "emerging" and gap["forecast_status"] == "ok"
    assert "outlook: emerging (+11%, unvalidated preview)" in gap["why"]


def test_learning_path_note_and_single_stage(preview_env):
    r = service.build_report(["skill:accounting", "skill:excel"])
    assert "heuristic" in r["learning_path"]["note"] and "not a curriculum" in r["learning_path"]["note"]
    assert [s["title"] for s in r["learning_path"]["stages"]] == ["Start here"]
    assert _s(r["learning_path"]["stages"][0]["skills"], "skill_id") == ["skill:python"]


def test_caveats_always_carry_the_disclaimer_and_preview_warning(preview_env):
    r = service.build_report(["skill:accounting", "skill:excel"])
    assert r["disclaimer"].startswith("Test disclaimer")
    assert r["caveats"][0].startswith("Test disclaimer")
    assert any(c.startswith("PREVIEW: outlook figures are unvalidated") for c in r["caveats"])
    assert any("alias matcher" in c for c in r["caveats"])


# ---- nothing within reach: two jobs, both below the reach threshold ------------------------------
def test_nothing_within_reach_shows_closest_jobs_and_their_gaps(preview_env):
    r = service.build_report(["skill:accounting"])
    assert r["status"] == "nothing_within_reach"
    assert "closest jobs" in r["message"]
    assert [j["job_id"] for j in r["best_fit_jobs"]] == ["job:accountant", "job:ai_engineer"]
    assert r["best_fit_jobs"][0]["coverage"] == pytest.approx(0.5 / 1.05, abs=1e-4)  # 0.4762
    assert r["best_fit_jobs"][1]["coverage"] == pytest.approx(0.1)
    assert r["targets"] == ["job:accountant", "job:ai_engineer"]
    # python: max over targets of 0.2 (accountant) and 0.45 (ai engineer, 9 of 9 postings); excel: 0.35.
    assert _s(r["skill_gaps"], "skill_id") == ["skill:python", "skill:excel"]
    python = r["skill_gaps"][0]
    assert python["score"] == pytest.approx(0.45)
    assert python["for_jobs"] == ["job:accountant", "job:ai_engineer"]
    assert [g["stage"] for g in r["skill_gaps"]] == ["Start here", "Next"]
    assert [s["title"] for s in r["learning_path"]["stages"]] == ["Start here", "Next"]


def test_no_overlap_when_no_skill_is_in_any_profile(preview_env):
    r = service.build_report(["skill:sql"])
    assert r["status"] == "no_overlap" and r["best_fit_jobs"] == [] and r["skill_gaps"] == []
    assert r["resolved_skills"][0]["status"] == "matched"


# ---- feed handling -------------------------------------------------------------------------------
def test_live_feed_has_no_banner_and_is_labelled_validated(live_model_env):
    r = service.build_report(["skill:accounting", "skill:excel"])
    assert r["feed"] == "live" and r["validated"] is True and r["banner"] is None
    best = r["best_fit_jobs"][0]
    assert best["growth"] == pytest.approx(math.exp(0.01) - 1) and best["direction"] == "stable"
    assert any(c.startswith("Outlook figures come from a validated model") for c in r["caveats"])
    assert not any(c.startswith("PREVIEW") for c in r["caveats"])


def test_no_model_and_no_preview_gives_no_growth_and_a_no_forecast_banner(no_model_env):
    r = service.build_report(["skill:accounting", "skill:excel"])
    assert r["feed"] == "none" and r["validated"] is False
    assert r["banner"].startswith("NO FORECASTS:")
    best = r["best_fit_jobs"][0]
    assert best["growth"] is None and best["forecast_status"] == "no_forecast"
    assert "no forecast available" in r["skill_gaps"][0]["why"]


def test_non_three_month_horizon_has_no_growth_and_a_warning(preview_env):
    r = service.build_report(["skill:accounting", "skill:excel"], horizon_months=6)
    assert r["best_fit_jobs"][0]["growth"] is None
    assert any("3-month horizon" in w for w in r["warnings"])


# ---- market snapshot is filtered to gap skills and stays descriptive -----------------------------
def test_market_snapshot_only_for_gap_skills(preview_env, monkeypatch):
    fake = {"items": [
        {"entity_id": "skill:python", "kind": "skill", "name": "Python", "percent_change": 175.0,
         "mean_share_first": 0.2, "mean_share_last": 0.55, "support": 50},
        {"entity_id": "skill:accounting", "kind": "skill", "name": "Accounting", "percent_change": 5.0,
         "mean_share_first": 0.3, "mean_share_last": 0.31, "support": 99},
    ]}
    monkeypatch.setattr(store, "snapshot", lambda kind=None, limit=50: fake)
    r = service.build_report(["skill:accounting", "skill:excel"])
    assert r["market_snapshot"] == [{"entity_id": "skill:python", "name": "Python", "percent_change": 175.0,
                                     "mean_share_first": 0.2, "mean_share_last": 0.55,
                                     "type": "descriptive_not_forecast"}]


def test_snapshot_failure_is_a_warning_not_an_error(preview_env, monkeypatch):
    def boom(kind=None, limit=50):
        raise FileNotFoundError("raw file missing")
    monkeypatch.setattr(store, "snapshot", boom)
    r = service.build_report(["skill:accounting", "skill:excel"])
    assert r["market_snapshot"] == [] and any("snapshot could not be computed" in w for w in r["warnings"])


# ---- inputs --------------------------------------------------------------------------------------
def test_unresolved_inputs_and_free_text_blob(preview_env):
    r = service.build_report(["skill:python", "Quantum basket weaving", "skill:not_a_skill"],
                             text="I used Excel and accounting daily. Also R.")
    assert r["unresolved"] == ["Quantum basket weaving", "skill:not_a_skill"]
    by_id = {x["skill_id"]: x for x in r["resolved_skills"] if x["skill_id"]}
    assert by_id["skill:python"]["method"] == "taxonomy_id"
    assert by_id["skill:excel"]["method"] == "text_alias" and by_id["skill:excel"]["input"].lower() == "excel"
    assert by_id["skill:accounting"]["method"] == "text_alias"
    assert "skill:r" not in by_id  # two-character aliases are ignored in free text


def test_free_text_list_item_resolves_by_alias(preview_env):
    r = service.build_report(["Accounting", "python3"])
    ids = {x["input"]: x["skill_id"] for x in r["resolved_skills"]}
    assert ids == {"Accounting": "skill:accounting", "python3": "skill:python"}
    assert all(x["method"] == "alias" for x in r["resolved_skills"])


def test_no_skill_resolved_is_an_explained_empty_report(preview_env):
    r = service.build_report(["Quantum basket weaving"])
    assert r["status"] == "no_skills_resolved" and r["best_fit_jobs"] == [] and r["skill_gaps"] == []
    assert r["unresolved"] == ["Quantum basket weaving"] and r["message"]
    assert r["learning_path"]["stages"] == []


def test_empty_input_does_not_raise(preview_env):
    r = service.build_report([])
    assert r["status"] == "no_skills_resolved" and r["resolved_skills"] == []


def test_input_limits_raise_value_error(preview_env):
    with pytest.raises(ValueError):
        service.build_report(["skill:python"] * (service.MAX_SKILLS + 1))
    with pytest.raises(ValueError):
        service.build_report([], text="x" * (service.MAX_TEXT_CHARS + 1))
    with pytest.raises(ValueError):
        service.build_report([], engine="gpt")


# ---- engine selection ----------------------------------------------------------------------------
def test_auto_without_database_url_uses_offline_and_says_so(preview_env):
    r = service.build_report(["skill:accounting"], engine="auto")
    assert r["engine"] == "offline_alias"
    assert not any("Qwen" in w for w in r["warnings"])


def test_explicit_qwen_without_database_url_falls_back_with_a_warning(preview_env):
    r = service.build_report(["skill:accounting"], engine="qwen_pgvector")
    assert r["engine"] == "offline_alias"
    assert any("DATABASE_URL is not set" in w and "offline alias matcher" in w for w in r["warnings"])


def test_real_engine_failure_falls_back_with_a_warning(preview_env, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
    r = service.build_report(["skill:accounting", "skill:excel"], engine="auto")
    assert r["engine"] == "offline_alias"
    assert any(w.startswith("The Qwen/pgvector matcher failed") and "offline alias matcher was used" in w
               for w in r["warnings"])
    assert r["best_fit_jobs"][0]["job_id"] == "job:accountant"


# ---- helpers and constants that must match the embeddings team --------------------------------------
@pytest.mark.parametrize("n, expected", [
    (1, ["Start here"]),
    (2, ["Start here", "Next"]),
    (3, ["Start here", "Next", "Later"]),
    (4, ["Start here", "Start here", "Next", "Later"]),
    (5, ["Start here", "Start here", "Next", "Next", "Later"]),
])
def test_stages_split_gaps_into_at_most_three_groups(n, expected):
    assert service._stages([{}] * n) == expected
    assert service._stages([]) == []


def _const(text: str, name: str) -> float:
    m = re.search(rf"^{name}\s*=\s*([0-9.]+)", text, re.M)
    assert m, f"{name} not found in embeddings source"
    return float(m.group(1))


def test_constants_match_the_embeddings_team_source():
    match_src = (ROOT / "embeddings" / "match.py").read_text(encoding="utf-8")
    graph_src = (ROOT / "embeddings" / "graph.py").read_text(encoding="utf-8")
    assert service.PROFILE_SIZE == _const(match_src, "PROFILE_SIZE")
    assert service.REACH == _const(match_src, "REACH")
    assert service.TOP_TARGETS == _const(match_src, "TOP_TARGETS")
    assert service.MIN_POSTINGS == _const(graph_src, "MIN_POSTINGS")
