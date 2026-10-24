"""JobStreet adapter with tiny hand-written rows (fixtures, not datasets)."""
import pandas as pd

from src.sources import jobstreet as js
from src.sources.common import source_cfg


def _raw(rows):
    return pd.DataFrame(rows, columns=["job_id", "job_title", "descriptions", "role", "listingDate"])


def _cfg(**overrides):
    return {**source_cfg("jobstreet_my"), **overrides}


ROWS = [  # weeks (Monday start): Jan 1 (2 rows), Jan 15 (2 rows), Feb 5 (1 row); Jan 8 is empty
    (1, "Python Developer", "python and sql", "Developer", "2024-01-03T00:00:00Z"),
    (2, "Accountant", "excel, gst", "Accountant", "2024-01-04T00:00:00Z"),
    (3, "Data Analyst", "power bi, sql", "Analyst", "2024-01-17T00:00:00Z"),
    (4, "Sales Executive", "sales", "Sales", "2024-01-18T00:00:00Z"),
    (5, "Accountant", "sql", "Accountant", "2024-02-05T00:00:00Z"),
]


def test_build_observations_weekly_shares():
    obs, meta = js.build_observations(js.prepare(_raw(ROWS), "weekly"), _cfg(min_period_postings=1, min_entity_total=1))
    sql = obs[(obs.entity_id == "skill:sql") & (obs.kind == "skill")].set_index("period")
    assert sql.loc[pd.Timestamp("2024-01-01"), "value"] == 0.5
    assert sql.loc[pd.Timestamp("2024-01-15"), "value"] == 0.5
    assert sql.loc[pd.Timestamp("2024-02-05"), "support"] == 1
    assert meta["dropped_periods"] == [] and meta["gap_periods"] == ["2024-01-08", "2024-01-22", "2024-01-29"]


def test_low_volume_period_dropped_and_gaps_not_zero_filled():
    obs, meta = js.build_observations(js.prepare(_raw(ROWS), "weekly"), _cfg(min_period_postings=2, min_entity_total=1))
    assert sorted(obs["period"].unique()) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-15")]
    assert [d["period"] for d in meta["dropped_periods"]] == ["2024-02-05"]
    assert meta["dropped_periods"][0]["total_postings"] == 1
    assert meta["gap_periods"] == ["2024-01-08"]  # the empty week between the kept ones; no rows, no zero-fill
    assert obs[obs["period"] == pd.Timestamp("2024-01-08")].empty
