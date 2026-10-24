"""Time alignment, loaders and the validator on tiny hand-written frames (fixtures, not datasets)."""
import pandas as pd

from src.sources import loaders
from src.sources.validate import validate
from src.training.features import build_feature_table, first_visible_period, shift_to_publication

WEEKLY_REG = {"sources": [
    {"source_id": "p", "role": "target", "granularity": "weekly", "reliability": 1.0},
    {"source_id": "m", "role": "leading", "enabled": True, "granularity": "monthly", "reliability": 0.5,
     "release_lag_days": 0, "max_staleness_days": 400},
    {"source_id": "w", "role": "leading", "enabled": True, "granularity": "weekly", "reliability": 0.5,
     "release_lag_days": 0, "max_staleness_days": 400},
]}
LAGGED_REG = {"sources": [
    WEEKLY_REG["sources"][0],
    {**WEEKLY_REG["sources"][1], "release_lag_days": 30},
]}


def _rows(entity, values, start, source, kind="skill", granularity_freq="W-MON"):
    periods = pd.date_range(start, periods=len(values), freq=granularity_freq)
    return pd.DataFrame({"entity_id": entity, "kind": kind, "period": periods, "source_id": source,
                         "value": values, "support": 50.0}, columns=loaders.COLUMNS)


def _obs(weeks=60):
    """Target 'p' weekly from 2023-01-02 (a Monday) plus monthly 'm' from Jan 2023 to Mar 2024."""
    target = _rows("a", [0.01 + 0.0005 * i for i in range(weeks)], "2023-01-02", "p")
    months = pd.date_range("2023-01-01", periods=15, freq="MS")
    m = pd.DataFrame({"entity_id": "a", "kind": "skill", "period": months, "source_id": "m",
                      "value": [0.02 + 0.001 * i for i in range(15)], "support": 40.0}, columns=loaders.COLUMNS)
    return pd.concat([target, m], ignore_index=True)


def test_monthly_figure_is_first_usable_in_the_week_after_month_end():
    starts = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]))
    # Jan is complete at 2024-02-01 (Thursday): the origin week 2024-01-29 ends 2024-02-05, the first to see it
    assert first_visible_period(starts, "monthly", 0).tolist() == [pd.Timestamp("2024-01-29"), pd.Timestamp("2024-02-26")]
    # with a 7-day release lag Jan is known at 2024-02-08, so the first origin is the week of 2024-02-05
    assert first_visible_period(starts.iloc[:1], "monthly", 7).tolist() == [pd.Timestamp("2024-02-05")]


def test_weekly_figure_is_usable_in_its_own_week():
    starts = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-08"]))
    assert first_visible_period(starts, "weekly", 0).tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")]


def test_shift_to_publication_keeps_native_period():
    obs = _obs(weeks=8)
    shifted = shift_to_publication(obs, WEEKLY_REG)
    m = shifted[shifted["source_id"] == "m"].iloc[0]
    assert m["source_period"] == pd.Timestamp("2023-01-01") and m["period"] == pd.Timestamp("2023-01-30")


def test_not_yet_visible_monthly_value_changes_no_feature():
    obs = _obs()
    as_of = "2024-01-22"  # the Jan 2024 figure is only visible from the week of 2024-01-29
    full = build_feature_table(obs, WEEKLY_REG, as_of=as_of)
    jan = (obs["source_id"] == "m") & (obs["period"] == "2024-01-01")
    pd.testing.assert_frame_equal(full, build_feature_table(obs[~jan], WEEKLY_REG, as_of=as_of))
    # sanity: once the figure is visible, removing it does change features (the check is not vacuous)
    later = build_feature_table(obs, WEEKLY_REG, as_of="2024-02-05")
    assert not later.equals(build_feature_table(obs[~jan], WEEKLY_REG, as_of="2024-02-05"))


def test_release_lag_delays_the_figure():
    obs = _obs()
    no_lag = build_feature_table(obs, WEEKLY_REG, as_of="2024-01-29").set_index("period")
    lagged = build_feature_table(obs, LAGGED_REG, as_of="2024-01-29").set_index("period")
    # with a 30-day lag the Jan figure is visible only from 2024-02-26; at 2024-01-29 it is not yet known
    assert "m__z" in no_lag.columns and "m__z" in lagged.columns
    assert no_lag.loc["2024-01-29", "m__z"] != lagged.loc["2024-01-29", "m__z"]


def test_mapped_and_global_scopes(tmp_path, monkeypatch):
    raw_dir, map_dir = tmp_path / "raw", tmp_path / "maps"
    raw_dir.mkdir(); map_dir.mkdir()
    monkeypatch.setattr(loaders, "RAW_DIR", raw_dir)
    monkeypatch.setattr(loaders, "MAPPING_DIR", map_dir)
    pd.DataFrame({"entity_id": ["ICT"], "kind": ["sector"], "period": ["2023-01-02"],
                  "source_id": ["g"], "value": [5.0], "support": [None]}).to_csv(raw_dir / "g.csv", index=False)
    pd.DataFrame({"entity_id": ["python", "sql"], "kind": ["skill", "skill"], "key": ["ICT", "ICT"]}).to_csv(map_dir / "g.csv", index=False)
    out = loaders.CsvLoader("g", scope="mapped").fetch()
    assert sorted(out["entity_id"]) == ["python", "sql"] and (out["value"] == 5.0).all()
    out = loaders.CsvLoader("g", scope="global").fetch()
    assert (out["entity_id"] == loaders.GLOBAL_ID).all()


def test_validator_flags_bad_rows_and_checks_periods():
    df = _rows("a", [0.1] * 60, "2022-01-03", "p")
    df.loc[0, "value"] = -1
    res = validate(pd.concat([df, df.iloc[[1]]]), {"source_id": "p", "granularity": "weekly"})
    assert not res["ok"] and any("negative" in e for e in res["errors"]) and any("duplicate" in e for e in res["errors"])
    assert validate(_rows("a", [0.1] * 60, "2022-01-03", "p"), {"source_id": "p", "granularity": "weekly"})["ok"]


def test_validator_period_thresholds_scale_with_granularity():
    monthly = _rows("a", [0.1] * 12, "2023-01-01", "m", granularity_freq="MS")
    res = validate(monthly, {"source_id": "m", "granularity": "monthly"})
    assert res["facts"]["periods"] == 12 and not any("only" in w for w in res["warnings"])
    weekly = validate(_rows("a", [0.1] * 12, "2023-01-02", "p"), {"source_id": "p", "granularity": "weekly"})
    assert any("only 12 distinct weekly periods" in w for w in weekly["warnings"])
    off_grid_df = pd.DataFrame({"entity_id": "a", "kind": "skill", "period": ["2023-01-03", "2023-01-10"],
                                "source_id": "p", "value": [0.1, 0.1], "support": 50.0})
    off_grid = validate(off_grid_df, {"source_id": "p", "granularity": "weekly"})
    assert any("not weekly period starts" in w for w in off_grid["warnings"])
