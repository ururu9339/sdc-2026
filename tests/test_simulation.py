import time

import numpy as np
import pandas as pd
import pytest

from src.sources.loaders import COLUMNS
from src.training import config as C
from src.training.audit import signal_audit
from src.validation import simulate as S


def test_generation_is_deterministic_for_a_seed():
    a = S.generate("S2", 1.0, seed=3, entities=12, periods=60)
    b = S.generate("S2", 1.0, seed=3, entities=12, periods=60)
    pd.testing.assert_frame_equal(a.obs, b.obs)
    np.testing.assert_array_equal(a.truth["latent_target"], b.truth["latent_target"])
    c = S.generate("S2", 1.0, seed=4, entities=12, periods=60)
    assert not np.allclose(a.obs["value"].to_numpy(), c.obs["value"].to_numpy())


def test_common_random_numbers_strength_only_changes_the_signal():
    lo = S.generate("S2", 0.0, seed=1, entities=12, periods=60).obs
    hi = S.generate("S2", 2.0, seed=1, entities=12, periods=60).obs
    b_lo = lo[lo["source_id"] == S.LEAD_B]["value"].to_numpy()
    b_hi = hi[hi["source_id"] == S.LEAD_B]["value"].to_numpy()
    np.testing.assert_array_equal(b_lo, b_hi)


def test_long_format_schema_and_consistency():
    ds = S.generate("S1", 1.0, seed=0, entities=10, periods=40)
    obs = ds.obs
    assert list(obs.columns) == COLUMNS
    assert len(obs) == 10 * 40 * 3
    assert set(obs["source_id"]) == {S.TARGET_ID, S.LEAD_A, S.LEAD_B}
    assert set(obs["kind"]) == {"skill", "job"}
    assert pd.api.types.is_datetime64_any_dtype(obs["period"])
    assert obs["value"].between(0, 1).all() and (obs["support"] >= 0).all()
    assert np.allclose(obs["support"], np.round(obs["support"]))

    for (sid, per), g in obs[obs["value"] > 0].groupby(["source_id", "period"]):
        totals = (g["support"] / g["value"]).round(6)
        assert totals.nunique() == 1, (sid, per)
    roles = {s["source_id"]: s["role"] for s in ds.registry["sources"]}
    assert roles == {S.TARGET_ID: "target", S.LEAD_A: "leading", S.LEAD_B: "leading"}
    reliab = {s["source_id"]: s["reliability"] for s in ds.registry["sources"]}
    assert reliab[S.LEAD_A] > reliab[S.LEAD_B]
    assert all(s["granularity"] == S.GRANULARITY[C.PERIOD_FREQ] for s in ds.registry["sources"])


def test_planted_leading_source_peaks_at_the_lead_lag():
    ds = S.generate("S2", 2.0, seed=0, entities=60, periods=130)
    lag = ds.truth["lead_lag"]
    latent = ds.truth["latent_target"]
    planted = ds.truth["lead_signal"]
    obs = ds.obs
    n, t_n = latent.shape

    def corr_at(x, j):
        a = x[:, : t_n - j]
        b = latent[:, j:]
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    lags = range(0, 16)
    exact = {j: corr_at(planted, j) for j in lags}
    assert max(exact, key=exact.get) == lag, exact
    assert exact[lag] > 0.999

    def panel(sid):
        p = obs[obs["source_id"] == sid].pivot(index="entity_id", columns="period", values="value")
        return np.log(p.loc[ds.truth["entity_ids"]].to_numpy())

    lead_a, lead_b = panel(S.LEAD_A), panel(S.LEAD_B)
    assert corr_at(lead_a, lag) > 0.5, corr_at(lead_a, lag)
    assert max(abs(corr_at(lead_b, j)) for j in lags) < 0.1


def test_noise_only_audit_dispersion_is_about_one():
    ds = S.generate("S0", 0.0, seed=0, entities=60, periods=130)
    audit = signal_audit(ds.obs, ds.registry)
    target = audit[audit["source_id"] == S.TARGET_ID]
    assert len(target) == 60
    assert 0.85 < target["dispersion_ratio"].median() < 1.15
    assert target["noise_like"].mean() > 0.9


def test_trend_scenario_is_not_noise_like_at_high_strength():
    ds = S.generate("S1", 2.0, seed=0, entities=60, periods=130)
    audit = signal_audit(ds.obs, ds.registry)
    target = audit[audit["source_id"] == S.TARGET_ID]
    assert target["dispersion_ratio"].median() > 1.3


def test_monthly_request_refused_when_the_pipeline_runs_weekly():
    other = "M" if C.PERIOD_FREQ == "W" else "W"
    with pytest.raises(ValueError):
        S.generate("S0", 0.0, seed=0, entities=6, periods=20, freq=other)


def test_outputs_refuse_pipeline_state_folders():
    for bad in (S.ROOT / "data", S.ROOT / "serving", S.ROOT / "src" / "models"):
        with pytest.raises(ValueError):
            S.write_outputs({}, bad)


def test_quick_end_to_end_run_is_fast_and_complete():
    t0 = time.perf_counter()
    res = S.run_simulation(seeds=1, entities=24, periods=110, strengths=(1.0,), workers=3)
    elapsed = time.perf_counter() - t0
    assert elapsed < 60, elapsed
    assert set(res) == {"meta", "aggregate", "runs"}
    assert {r["scenario"] for r in res["aggregate"]} == {"S0", "S1", "S2"}
    assert len(res["runs"]) == 1 + 1 + 1 + 1
    assert all(r.get("noise_floor") is not None for r in res["runs"] if r["scenario"] != "S3")
    for r in res["runs"]:
        if r["scenario"] == "S3":
            assert r["truth_duplicate_groups"] >= 0
            continue
        assert set(r["gates"]) == {"production", "demo"}
        for p in r["gates"]:
            assert isinstance(r["gates"][p]["passed"], bool)
        assert set(r["importance"]["share"]) >= {S.TARGET_ID, S.LEAD_A, S.LEAD_B, "composite", "calendar_meta"}
        assert set(r["ablation"]) == set(S.CONFIGS)
    md = S.render_markdown(res)
    assert "SYNTHETIC" in md.splitlines()[2] and "Malaysian" in md
