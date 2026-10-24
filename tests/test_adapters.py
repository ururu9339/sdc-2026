"""Job-descriptions and job-market adapters and period selection on tiny hand-written CSVs (fixtures, not datasets).

Adapter parameters come from the registry entry; tests pass an overridden copy instead of editing module globals."""
import json

import pandas as pd
import pytest

from src.sources import job_descriptions as jd
from src.sources import job_market as jm
from src.sources.common import gap_periods, select_periods, source_cfg


def _cfg(source_id, **overrides):
    return {**source_cfg(source_id), **overrides}


@pytest.fixture
def jd_csv(tmp_path):
    rows = [  # weeks (Monday start): Jan 1, Jan 8, Jan 8, Jan 15, -, Jan 22, Jan 29, Jan 29
        ("2024-01-05", "Data Analyst", "Data Analyst", "python, sql"),
        ("2024-01-09", "Accountant", "", "excel"),
        ("2024-01-10", "Random Thing", "Random", "python sql"),
        ("2024-01-20", "Data Analyst", "Analyst", None),  # Role has no family: falls back to Job Title
        ("not-a-date", "Data Analyst", "Data Analyst", "python"),  # unparseable date: excluded
        ("2024-01-23", "Accountant", "Accountant", "sql"),
        ("2024-02-03", "Data Analyst", "Data Analyst", "python"),
        ("2024-02-04", "Accountant", "Accountant", "sql"),
    ]
    path = tmp_path / "jd.csv"
    pd.DataFrame(rows, columns=["Job Posting Date", "Job Title", "Role", "skills"]).to_csv(path, index=False)
    return path


def test_jd_weekly_shares_graph_and_meta(tmp_path, jd_csv):
    out, graph = tmp_path / "job_postings.csv", tmp_path / "graph.csv"
    jd.build(jd_csv, out=out, graph_out=graph, cfg=_cfg("job_postings", min_period_postings=1, min_entity_total=1),
             chunk_rows=2)

    obs = pd.read_csv(out, parse_dates=["period"]).set_index(["entity_id", "period"])
    wk = lambda d: pd.Timestamp(d)  # noqa: E731
    assert obs.loc[("skill:python", wk("2024-01-01")), "value"] == 1.0  # row 1 of 1 postings
    assert obs.loc[("skill:python", wk("2024-01-08")), "value"] == 0.5  # row 3 of 2 postings
    assert obs.loc[("skill:python", wk("2024-01-29")), "value"] == 0.5  # row 7 of 2
    assert obs.loc[("job:data_analyst", wk("2024-01-08")), "value"] == 0.0  # present entity, zero that week
    assert obs.loc[("job:accountant", wk("2024-01-08")), "value"] == 0.5  # row 2 of 2 (Role matched)
    assert obs.loc[("job:accountant", wk("2024-01-22")), "support"] == 1
    assert obs.loc[("job:data_analyst", wk("2024-01-15")), "value"] == 1.0  # Role empty -> Job Title fallback
    assert obs["kind"].isin(["skill", "job"]).all()

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["granularity"] == "weekly" and meta["first_period"] == "2024-01-01" and meta["last_period"] == "2024-01-29"
    assert meta["rows_read"] == 8 and meta["rows_used"] == 7
    assert [p["total_postings"] for p in meta["periods"]] == [1, 2, 1, 1, 2]
    assert meta["dropped_periods"] == []

    edges = pd.read_csv(graph).set_index(["job", "skill"])
    # data_analyst: rows 1, 4, 7 are dated and matched (the bad-date row is excluded)
    assert edges.loc[("job:data_analyst", "skill:python"), "n_job_postings"] == 3
    assert edges.loc[("job:data_analyst", "skill:python"), "n_cooccur"] == 2
    assert edges.loc[("job:data_analyst", "skill:python"), "weight"] == pytest.approx(2 / 3)
    assert edges.loc[("job:data_analyst", "skill:sql"), "weight"] == pytest.approx(1 / 3)
    assert edges.loc[("job:accountant", "skill:excel"), "weight"] == pytest.approx(1 / 3)


def test_jd_chunking_does_not_change_output(tmp_path, jd_csv):
    cfg = _cfg("job_postings", min_period_postings=1, min_entity_total=1)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    jd.build(jd_csv, out=a, graph_out=tmp_path / "ga.csv", cfg=cfg, chunk_rows=2)
    jd.build(jd_csv, out=b, graph_out=tmp_path / "gb.csv", cfg=cfg, chunk_rows=1000)
    pd.testing.assert_frame_equal(pd.read_csv(a), pd.read_csv(b))
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "ga.csv"), pd.read_csv(tmp_path / "gb.csv"))


def test_dense_source_with_an_empty_week_is_an_error(tmp_path, jd_csv):
    # with a threshold of 2 the Jan 15 week (1 posting) is dropped, leaving a hole in a dense series
    with pytest.raises(ValueError, match="dense range"):
        jd.build(jd_csv, out=tmp_path / "x.csv", graph_out=tmp_path / "g.csv",
                 cfg=_cfg("job_postings", min_period_postings=2, min_entity_total=1))


def test_select_periods_drops_low_volume_and_respects_density():
    weeks = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"])
    totals = pd.Series([2, 10, 10, 3], index=weeks)
    grid, dropped = select_periods(totals, 3, "weekly", dense=True)
    assert list(grid) == list(weeks[1:]) and dropped == [
        {"period": "2024-01-01", "total_postings": 2, "reason": "fewer than 3 postings"}]
    sparse = pd.Series([10, 10, 10], index=pd.to_datetime(["2024-01-01", "2024-01-15", "2024-01-29"]))
    grid, _ = select_periods(sparse, 3, "weekly", dense=False)
    assert len(grid) == 3  # gaps are kept as gaps, not filled
    assert gap_periods(grid, "weekly") == ["2024-01-08", "2024-01-22"]
    with pytest.raises(ValueError):
        select_periods(sparse, 3, "weekly", dense=True)


def test_job_market_shares_and_zero_fill(tmp_path):
    raw = pd.DataFrame({
        "year": [2024] * 3 + [2024] + [2024] * 2 + [2024] * 2,
        "month": [1] * 3 + [1] + [2] * 2 + [2] * 2,
        "occupation": ["UX Designer"] * 3 + ["Data Analyst"] + ["Data Analyst"] * 2 + ["Unknown Role"] * 2,
    })
    cfg = _cfg("market_signal", min_period_postings=1, min_entity_total=1)
    obs, meta = jm.build_observations(raw, cfg)
    obs = obs.set_index(["entity_id", "period"])
    assert obs.loc[("job:ux_designer", "2024-01-01"), "value"] == 0.75  # 3 of 4 January rows
    assert obs.loc[("job:data_analyst", "2024-01-01"), "value"] == 0.25
    assert obs.loc[("job:data_analyst", "2024-02-01"), "value"] == 0.5  # 2 of 4 February rows (unknown ones count)
    assert obs.loc[("job:ux_designer", "2024-02-01"), "value"] == 0  # contiguous month with no rows: zero
    assert set(obs.index.get_level_values("entity_id")) == {"job:ux_designer", "job:data_analyst"}
    assert meta["unmatched_occupations"] == ["Unknown Role"]
    assert obs["kind"].eq("job").all()

    raw.to_csv(tmp_path / "market.csv", index=False)
    out = jm.build(tmp_path / "market.csv", out=tmp_path / "market_signal.csv", cfg=cfg)
    written = pd.read_csv(out)
    assert set(written.columns) == {"entity_id", "kind", "period", "source_id", "value", "support"}
    assert (written["source_id"] == "market_signal").all()
    assert out.with_suffix(".meta.json").exists()
