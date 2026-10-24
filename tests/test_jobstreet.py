"""Mechanics test with a few hand-written rows (a fixture, not a dataset)."""
import pandas as pd

from src.sources import jobstreet as js


def test_build_observations_shares(tmp_path):
    df = pd.DataFrame({
        "job_id": [1, 2, 3, 4],
        "job_title": ["Python Developer", "Accountant", "Data Analyst", "Sales Executive"],
        "descriptions": ["python and sql", "excel, gst", "power bi, sql", "sales"],
        "role": ["Developer", "Accountant", "Analyst", "Sales"],
        "listingDate": ["2024-01-05T00:00:00Z", "2024-01-20T00:00:00Z", "2024-02-02T00:00:00Z", "2024-02-03T00:00:00Z"],
    })
    df.to_csv(tmp_path / "t.csv", index=False)
    df = js.load_raw(tmp_path / "t.csv")
    js.MIN_SKILL_POSTINGS = 1
    js.MIN_ROLE_POSTINGS = 1
    obs = js.build_observations(df)
    sql = obs[(obs.entity_id == "sql") & (obs.kind == "skill")].set_index("period")
    assert sql.loc["2024-01-01", "value"] == 0.5 and sql.loc["2024-02-01", "support"] == 1
