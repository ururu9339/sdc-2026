import json

import numpy as np
import pandas as pd
import pytest

from src.pipeline import agent_loop as AL
from src.pipeline.store import AppendOnlyLog
from src.sources.loaders import COLUMNS
from src.training import config as C
from src.training.audit import duplicate_groups, duplicate_summary, non_canonical_ids
from src.training.features import build_feature_table, feature_columns, usable
from src.training.forecast import DUPLICATE_CAVEAT, make_forecasts
from src.training.train import fit_models

REG = {"version": 5, "disclaimer": "test disclaimer", "sources": [
    {"source_id": "p", "role": "target", "enabled": True, "reliability": 0.9, "granularity": "weekly",
     "release_lag_days": 0, "loader": {"type": "csv"}},
    {"source_id": "w", "role": "leading", "enabled": True, "reliability": 0.5, "granularity": "weekly",
     "release_lag_days": 0, "max_staleness_days": 400, "loader": {"type": "csv"}},
]}
WEEKS = pd.date_range("2023-01-02", periods=40, freq="W-MON")
PATH = [0.02 + 0.0004 * i + 0.003 * np.sin(i / 3.0) for i in range(40)]


def _series(ent, values, kind="skill", source="p", n_from=0):
    periods = WEEKS[n_from:n_from + len(values)]
    return pd.DataFrame({"entity_id": ent, "kind": kind, "period": periods, "source_id": source,
                         "value": values, "support": 100.0}, columns=COLUMNS)


def _obs(*frames):
    return pd.concat(frames, ignore_index=True)


def test_identical_series_form_one_group_with_the_smallest_id_as_canonical():
    obs = _obs(_series("skill:swift", PATH), _series("skill:kotlin", PATH), _series("skill:mobile_dev", PATH),
               _series("skill:python", [0.05 - 0.001 * i for i in range(40)]))
    groups = duplicate_groups(obs, REG)
    assert len(groups) == 1
    g = groups[0]
    assert g["canonical"] == "skill:kotlin"
    assert g["members"] == ["skill:kotlin", "skill:mobile_dev", "skill:swift"]
    assert g["shared_periods"] == 40 and g["kinds"] == ["skill"]


def test_near_identical_series_are_not_duplicates():
    nudged = [v * (1 + 1e-4) for v in PATH]
    assert duplicate_groups(_obs(_series("skill:aws", PATH), _series("skill:azure", nudged)), REG) == []


def test_fewer_than_min_periods_shared_is_not_a_duplicate():

    twelve = _series("skill:azure", PATH[:12])
    assert C.DEDUP["min_periods"] == 13
    assert duplicate_groups(_obs(_series("skill:aws", PATH), twelve), REG) == []


def test_all_zero_series_do_not_match_each_other():
    zeros = [0.0] * 20
    assert duplicate_groups(_obs(_series("skill:a1", zeros), _series("skill:b1", zeros)), REG) == []


def test_only_the_target_source_counts():
    obs = _obs(_series("skill:aws", PATH), _series("skill:azure", PATH), _series("skill:aws", PATH, source="w"),
               _series("skill:azure", PATH, source="w"))
    groups = duplicate_groups(obs, REG)
    assert [g["members"] for g in groups] == [["skill:aws", "skill:azure"]]


def test_duplicates_are_judged_only_on_data_known_at_as_of():

    obs = _obs(_series("skill:aws", PATH), _series("skill:azure", PATH))
    assert duplicate_groups(obs, REG, as_of=WEEKS[10]) == []
    assert len(duplicate_groups(obs, REG, as_of=WEEKS[30])) == 1


def test_no_duplicates_gives_an_empty_list_and_a_zero_summary():
    obs = _obs(_series("skill:aws", PATH), _series("skill:python", [0.05 - 0.001 * i for i in range(40)]))
    groups = duplicate_groups(obs, REG)
    assert groups == []
    assert duplicate_summary(groups, n_entities=2) == {"groups": 0, "entities_in_groups": 0,
                                                       "entities_collapsed": 0, "series_before": 2,
                                                       "series_after": 2}


def test_summary_counts_groups_and_collapsed_entities():
    other = [0.03 + 0.0002 * i for i in range(40)]
    obs = _obs(_series("skill:kotlin", PATH), _series("skill:mobile_dev", PATH), _series("skill:swift", PATH),
               _series("skill:aws", other), _series("skill:azure", other), _series("skill:sql", [0.01] * 40))
    groups = duplicate_groups(obs, REG)
    s = duplicate_summary(groups, n_entities=6)
    assert s == {"groups": 2, "entities_in_groups": 5, "entities_collapsed": 3, "series_before": 6, "series_after": 3}
    assert non_canonical_ids(groups) == {"skill:mobile_dev", "skill:swift", "skill:azure"}


@pytest.fixture
def dup_obs():
    return _obs(_series("skill:kotlin", PATH), _series("skill:mobile_dev", PATH), _series("skill:swift", PATH),
                _series("skill:python", [0.05 - 0.001 * i + 0.002 * np.cos(i) for i in range(40)]))


def test_non_canonical_members_are_dropped_from_the_feature_table(dup_obs):
    as_of = WEEKS[-1]
    feats = build_feature_table(dup_obs, REG, as_of)
    assert set(feats["entity_id"]) == {"skill:kotlin", "skill:python"}
    everything = build_feature_table(dup_obs, REG, as_of, dedup=False)
    assert set(everything["entity_id"]) == {"skill:kotlin", "skill:mobile_dev", "skill:swift", "skill:python"}


def test_every_member_is_forecast_and_copies_the_canonical_numbers(dup_obs):
    as_of = WEEKS[-1]
    groups = duplicate_groups(dup_obs, REG, as_of)
    feats = build_feature_table(dup_obs, REG, as_of)
    cols = feature_columns(feats)
    train = usable(feats)
    train = train[train["y"].notna()]
    models = fit_models(train, cols)
    meta = {"version": "v1", "feature_columns": cols, "gate_profile": "demo", "horizon_periods": C.HORIZON}
    out = make_forecasts(models, meta, feats, as_of, REG, duplicates=groups)
    by = {r["entity_id"]: r for r in out}
    assert set(by) == {"skill:kotlin", "skill:mobile_dev", "skill:swift", "skill:python"}
    members = ["skill:kotlin", "skill:mobile_dev", "skill:swift"]
    kot = by["skill:kotlin"]
    assert kot["status"] == "ok" and kot["duplicate_group"] == members and kot["shares_signal_with"] is None
    for m in ("skill:mobile_dev", "skill:swift"):
        rec = by[m]
        assert rec["status"] == kot["status"]
        assert rec["p50"] == pytest.approx(kot["p50"]) and rec["p10"] == pytest.approx(kot["p10"])
        assert rec["p90"] == pytest.approx(kot["p90"]) and rec["direction"] == kot["direction"]
        assert rec["shares_signal_with"] == "skill:kotlin"
        assert rec["duplicate_group"] == members
        assert rec["confidence"] == "low"
        assert rec["caveat"] == DUPLICATE_CAVEAT.format(canon="skill:kotlin")
        assert rec["source_contributions"] is None
    py = by["skill:python"]
    assert "shares_signal_with" not in py and "duplicate_group" not in py


def test_duplicate_copies_are_not_scored_twice_in_reconciliation(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    paths = {"DATA_DIR": data, "OBSERVATIONS_PATH": data / "obs.csv", "REGISTRY_PATH": tmp_path / "reg.json",
             "PREVIEW_LOG_PATH": data / "preview_log.jsonl", "PREVIEW_OUTCOME_LOG_PATH": data / "preview_out.jsonl",
             "FORECAST_LOG_PATH": data / "forecast_log.jsonl", "OUTCOME_LOG_PATH": data / "outcomes.jsonl",
             "RUN_LOG_PATH": data / "runs.jsonl", "VERSION_LOG_PATH": tmp_path / "version_log.json"}
    for k, v in paths.items():
        monkeypatch.setattr(C, k, v)
    (tmp_path / "reg.json").write_text(json.dumps(REG))
    as_of, target = WEEKS[20], WEEKS[32]
    base = {"model_version": "preview:v1", "gate_profile": "demo", "as_of": C.period_label(as_of),
            "target_period": C.period_label(target), "horizon_periods": 12, "status": "ok", "reason": None,
            "p50": 0.01, "p10": -0.05, "p90": 0.07, "baseline_trend": 0.0, "direction": "stable",
            "feed": "preview"}
    recs = [{**base, "forecast_id": "preview:v1:skill:kotlin", "entity_id": "skill:kotlin"},
            {**base, "forecast_id": "preview:v1:skill:swift", "entity_id": "skill:swift",
             "shares_signal_with": "skill:kotlin"}]
    AppendOnlyLog(C.PREVIEW_LOG_PATH).append_many(recs)
    feats = pd.DataFrame({"entity_id": ["skill:kotlin"] * 2, "period": [as_of, target],
                          "lvl": [np.log(0.03), np.log(0.035)]})
    loop = AL.AgentLoop()
    out = loop.reconcile(feats, target, feed="preview")
    assert [r["entity_id"] for r in out] == ["skill:kotlin"]
