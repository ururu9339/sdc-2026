"""Job-SDF adapter on a tiny hand-written parquet dataset in tmp_path (not the real dataset).

The fixture follows the identities the real data satisfies: proportion = demand / job-ad count per month,
r0 = the all-occupation aggregate, r0 demand = sum of r1 demand over occupations."""
import json

import numpy as np
import pandas as pd
import pytest

from src.sources import job_sdf as js
from src.sources.validate import validate

MONTHS = ["2021-01", "2021-02", "2021-03"]
ADS = {0: [10, 20, 30], 1: [5, 5, 5]}  # job ads per occupation per month
# mentions[occupation][skill] -> counts per month (skill 0 is common, skill 1 rare, skill 2 zero in month 2)
MENTIONS = {
    0: {0: [4, 6, 9], 1: [0, 1, 0], 2: [2, 0, 3]},
    1: {0: [1, 2, 2], 1: [0, 0, 1], 2: [1, 0, 0]},
}


def _write_dataset(root, mentions=MENTIONS, ads=ADS, r0_proportion_scale=1.0):
    """Writes demand/ and proportion/ tables for r0 and r1 under root and returns root."""
    skills = sorted(next(iter(mentions.values())).keys())
    r1_rows, r1_prop = [], []
    for o in sorted(mentions):
        for s in skills:
            d = mentions[o][s]
            r1_rows.append({"r1_id": o, "skill_id": s, **dict(zip(MONTHS, d))})
            r1_prop.append({"r1_id": o, "skill_id": s, **{m: d[i] / ads[o][i] for i, m in enumerate(MONTHS)}})
    r1 = pd.DataFrame(r1_rows)
    r1p = pd.DataFrame(r1_prop)
    total_ads = [sum(ads[o][i] for o in ads) for i in range(len(MONTHS))]
    r0_rows, r0_prop = [], []
    for s in skills:
        d = [sum(mentions[o][s][i] for o in mentions) for i in range(len(MONTHS))]
        r0_rows.append({"r0_id": 0, "skill_id": s, **dict(zip(MONTHS, d))})
        r0_prop.append({"r0_id": 0, "skill_id": s,
                        **{m: r0_proportion_scale * d[i] / total_ads[i] for i, m in enumerate(MONTHS)}})
    r0, r0p = pd.DataFrame(r0_rows), pd.DataFrame(r0_prop)
    for sub in ("demand", "proportion"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    r0.to_parquet(root / "demand" / "r0.parquet", index=False)
    r1.to_parquet(root / "demand" / "r1.parquet", index=False)
    r0p.to_parquet(root / "proportion" / "r0.parquet", index=False)
    r1p.to_parquet(root / "proportion" / "r1.parquet", index=False)
    return root


@pytest.fixture
def dataset(tmp_path):
    return _write_dataset(tmp_path / "dataset")


PROOF_REGISTRY = js.C.ROOT / "src" / "sources" / "registries" / "job_sdf_proof.json"


def _cfg(**overrides):
    reg = json.loads(PROOF_REGISTRY.read_text())
    return {**next(s for s in reg["sources"] if s["source_id"] == "job_sdf"), **overrides}


def test_inspect_identities_and_totals(dataset):
    facts = js.inspect(dataset)
    assert facts["identities_ok"] is True
    assert facts["total_ads"] == 75  # 15 + 25 + 35 job ads over the three months
    assert facts["occupations"] == 2 and facts["skills_in_r0"] == 3


def test_build_value_is_share_of_ads_and_support_is_count(dataset, tmp_path):
    cfg = _cfg(min_period_postings=1)
    out = js.build(dataset, out=tmp_path / "job_sdf.csv", cfg=cfg, top_skills=10, min_skill_total=1)
    obs = pd.read_csv(out, parse_dates=["period"]).set_index(["entity_id", "period"])
    jan, feb, mar = (pd.Timestamp(m) for m in ("2021-01-01", "2021-02-01", "2021-03-01"))

    # all-occupation ads per month: 15, 25, 35
    assert obs.loc[("skill_0", jan), "support"] == 5  # mentions of skill 0 in January: 4 + 1
    assert obs.loc[("skill_0", jan), "value"] == pytest.approx(5 / 15)
    assert obs.loc[("skill_0", mar), "value"] == pytest.approx((9 + 2) / 35)
    # an occupation's value is its share of all ads; support is its ad count
    assert obs.loc[("occupation_0", feb), "support"] == 20
    assert obs.loc[("occupation_0", feb), "value"] == pytest.approx(20 / 25)
    assert obs.loc[("occupation_1", jan), "value"] == pytest.approx(5 / 15)


def test_build_is_contiguous_zero_filled_and_long_format(dataset, tmp_path):
    cfg = _cfg(min_period_postings=1)
    out = js.build(dataset, out=tmp_path / "job_sdf.csv", cfg=cfg, top_skills=10, min_skill_total=1)
    obs = pd.read_csv(out, parse_dates=["period"])
    assert set(obs.columns) == {"entity_id", "kind", "period", "source_id", "value", "support"}
    assert (obs["source_id"] == "job_sdf").all()
    assert set(obs["kind"]) == {"skill", "job"}
    assert obs.groupby("entity_id")["period"].nunique().eq(3).all()  # 3 contiguous months, no gaps
    skill2 = obs[(obs["entity_id"] == "skill_2") & (obs["period"] == "2021-02-01")].iloc[0]
    assert skill2["support"] == 0 and skill2["value"] == 0  # zero-filled, not missing
    assert obs.loc[obs["entity_id"].str.startswith("occupation"), "kind"].eq("job").all()
    assert obs.loc[obs["entity_id"].str.startswith("skill"), "kind"].eq("skill").all()
    assert obs["value"].between(0, 1).all()


def test_meta_documents_the_denominator_and_checks(dataset, tmp_path):
    out = js.build(dataset, out=tmp_path / "job_sdf.csv", cfg=_cfg(min_period_postings=1),
                   top_skills=10, min_skill_total=1)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["granularity"] == "monthly" and meta["first_period"] == "2021-01-01"
    assert meta["last_period"] == "2021-03-01"
    assert [p["total_postings"] for p in meta["periods"]] == [15, 25, 35]
    assert "job advertisements" in meta["denominator_definition"]
    assert "Not a mention count" in meta["denominator_definition"]
    assert meta["identities"]["identities_ok"] is True
    assert meta["kept_entities"] == {"skill": 3, "job": 2}


def test_top_and_low_support_filters_are_reported(dataset, tmp_path):
    # totals over all months: skill_0 = 24, skill_2 = 6, skill_1 = 2. Top 2 = skill_0 and skill_2;
    # skill_2 is then below the minimum of 20 and is reported as dropped. skill_1 is outside the top 2.
    out = js.build(dataset, out=tmp_path / "a.csv", cfg=_cfg(min_period_postings=1),
                   top_skills=2, min_skill_total=20)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    obs = pd.read_csv(out)
    assert set(obs.loc[obs["kind"] == "skill", "entity_id"]) == {"skill_0"}
    assert meta["skills_dropped_low_total"] == [{"skill": "skill_2", "total_mentions": 6}]
    assert set(obs.loc[obs["kind"] == "job", "entity_id"]) == {"occupation_0", "occupation_1"}


def test_low_support_skills_are_dropped(dataset, tmp_path):
    out = js.build(dataset, out=tmp_path / "b.csv", cfg=_cfg(min_period_postings=1), top_skills=10,
                   min_skill_total=1000)
    obs = pd.read_csv(out)
    assert not (obs["kind"] == "skill").any()  # every skill is below 1000 mentions in this tiny fixture
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert len(meta["skills_dropped_low_total"]) == 3


def test_output_passes_the_source_contract(dataset, tmp_path):
    out = js.build(dataset, out=tmp_path / "job_sdf.csv", cfg=_cfg(min_period_postings=1),
                   top_skills=10, min_skill_total=1)
    res = validate(pd.read_csv(out), _cfg(min_period_postings=1), None)
    assert res["ok"], res["errors"]
    assert res["facts"]["granularity"] == "monthly" and res["facts"]["periods"] == 3


def test_inconsistent_proportion_is_rejected(tmp_path):
    root = _write_dataset(tmp_path / "bad", r0_proportion_scale=1.01)  # 1% off the demand / ads identity
    with pytest.raises(ValueError, match="integers|not demand / job-ad count"):  # 1% off breaks the integer check first
        js.inspect(root)


def test_occupations_must_sum_to_the_total(tmp_path):
    root = _write_dataset(tmp_path / "bad2")
    r1 = pd.read_parquet(root / "demand" / "r1.parquet")
    r1.loc[0, "2021-01"] += 3  # changes r1 demand, so r0 no longer equals the sum over occupations
    r1.to_parquet(root / "demand" / "r1.parquet", index=False)
    with pytest.raises(ValueError):
        js.inspect(root)


def test_mismatched_tables_are_rejected(tmp_path):
    root = _write_dataset(tmp_path / "bad3")
    p0 = pd.read_parquet(root / "proportion" / "r0.parquet")
    p0.loc[0, "skill_id"] = 99
    p0.to_parquet(root / "proportion" / "r0.parquet", index=False)
    with pytest.raises(ValueError, match="do not have the same rows"):
        js.inspect(root)


def test_registry_entry_points_at_the_adapter():
    reg = json.loads(PROOF_REGISTRY.read_text())
    assert reg["version"] == 5 and reg["disclaimer"]
    assert [s["source_id"] for s in reg["sources"]] == ["job_sdf"]
    src = reg["sources"][0]
    assert src["role"] == "target" and src["granularity"] == "monthly" and src["adapter"] == "src.sources.job_sdf"
    assert "CC BY-NC-SA" in src["license"] and "NeurIPS 2024" in src["attribution"]
    assert np.isclose(src["reliability"], 0.9) and src["tier"] == 1
