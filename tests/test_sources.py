"""Mechanics tests with tiny hand-written frames (fixtures, not datasets)."""
import pandas as pd

from src.sources import loaders
from src.sources.validate import validate
from src.training.features import build_feature_table

REG = {"sources": [
    {"source_id": "p", "role": "target", "reliability": 1.0},
    {"source_id": "g", "role": "leading", "enabled": True, "reliability": 0.8, "release_lag_months": 12, "ffill_limit": 12},
]}


def _obs(months=24):
    rows = []
    for m in pd.date_range("2022-01-01", periods=months, freq="MS"):
        rows.append(("a", "skill", m, "p", 0.01, 50.0))
    for y in (2022, 2023):
        rows.append(("a", "skill", pd.Timestamp(f"{y}-01-01"), "g", 100.0 * (y - 2021), float("nan")))
    return pd.DataFrame(rows, columns=loaders.COLUMNS)


def test_release_lag_blocks_unpublished_figure():
    f = build_feature_table(_obs(), REG, as_of="2022-12-01")
    assert "g__z" not in f.columns  # 2022 figure is not published until 2023-01, so the source is skipped
    # the 2023 figure is published 2024-01, so removing it must not change any feature at 2023-12
    full = build_feature_table(_obs(), REG, as_of="2023-12-01")
    obs = _obs()
    no_2023 = obs[~((obs["source_id"] == "g") & (obs["period"] == "2023-01-01"))]
    pd.testing.assert_frame_equal(full, build_feature_table(no_2023, REG, as_of="2023-12-01"))


def test_mapped_and_global_scopes(tmp_path, monkeypatch):
    raw_dir, map_dir = tmp_path / "raw", tmp_path / "maps"
    raw_dir.mkdir(); map_dir.mkdir()
    monkeypatch.setattr(loaders, "RAW_DIR", raw_dir)
    monkeypatch.setattr(loaders, "MAPPING_DIR", map_dir)
    pd.DataFrame({"entity_id": ["ICT"], "kind": ["sector"], "period": ["2023-01-01"],
                  "source_id": ["g"], "value": [5.0], "support": [None]}).to_csv(raw_dir / "g.csv", index=False)
    pd.DataFrame({"entity_id": ["python", "sql"], "kind": ["skill", "skill"], "key": ["ICT", "ICT"]}).to_csv(map_dir / "g.csv", index=False)
    out = loaders.CsvLoader("g", scope="mapped").fetch()
    assert sorted(out["entity_id"]) == ["python", "sql"] and (out["value"] == 5.0).all()
    out = loaders.CsvLoader("g", scope="global").fetch()
    assert (out["entity_id"] == loaders.GLOBAL_ID).all()


def test_validator_flags_bad_rows():
    df = _obs()
    df.loc[0, "value"] = -1
    res = validate(pd.concat([df, df.iloc[[1]]]))
    assert not res["ok"] and any("negative" in e for e in res["errors"]) and any("duplicate" in e for e in res["errors"])
    assert validate(_obs())["ok"]
