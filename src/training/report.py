from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from src.sources.loaders import get_loader
from src.training import config as C
from src.training.ablation import run_ablation
from src.training.audit import (duplicate_groups, duplicate_summary, records, signal_audit, source_coverage,
                                summarise_audit)
from src.training.backtest import evaluate_gate, walk_forward
from src.training.features import build_feature_table, column_source_map, feature_columns, usable
from src.training.train import fit_models, source_contributions


def load_observations(registry: dict) -> pd.DataFrame:
    frames = [get_loader(s).fetch() for s in C.enabled_sources(registry)]
    return pd.concat(frames, ignore_index=True)


def _verdict(gate: dict) -> str:
    note = C.GATE_PROFILES[gate["profile"]]["note"]
    head = f"GATE {'PASSED' if gate['passed'] else 'REJECTED'} under the {gate['profile'].upper()} profile ({note})"
    if "model_mae" not in gate:
        return f"{head}: {gate['reason']}"
    k = gate["constraints"]
    failed = [c for c, v in gate["checks"].items() if not v]
    why = {
        "beats_baseline": f"model MAE {gate['model_mae']:.4f} is not at least "
                          f"{k['mae_margin']:.0%} below the best baseline '{gate['best_baseline']}' "
                          f"({gate['best_baseline_mae']:.4f}); improvement {gate['improvement_pct']:+.1f}%",
        "fold_win_rate": f"model beat the baseline in {gate['fold_win_rate']:.0%} of folds "
                         f"(need {k['min_fold_win_rate']:.0%})",
        "calibrated": f"10-90% interval coverage {gate['coverage']:.2f} is outside {tuple(k['coverage_range'])}",
    }
    lines = [f"{head}: " + "; ".join(why[c] for c in failed)] if failed else [f"{head}: all checks passed"]
    lines.append(f"  folds={gate['n_folds']}, model MAE={gate['model_mae']:.4f}, "
                 f"best baseline '{gate['best_baseline']}'={gate['best_baseline_mae']:.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", help="YYYY-MM-DD (snapped to its week); default: last week of the target series")
    ap.add_argument("--profile", choices=sorted(C.GATE_PROFILES), help="default: the active profile")
    a = ap.parse_args()
    profile = C.GATE_PROFILES[a.profile] if a.profile else C.active_gate_profile()
    registry = C.load_registry()
    obs = load_observations(registry)
    tgt = C.target_source(registry)
    target_last = pd.to_datetime(obs.loc[obs["source_id"] == tgt, "period"]).max()
    as_of = min(C.to_period_start(pd.Timestamp(a.as_of)), target_last) if a.as_of else target_last

    audit = signal_audit(obs, registry, as_of)
    audit_summary = summarise_audit(audit, registry)
    groups = duplicate_groups(obs, registry, as_of)
    dup_summary = duplicate_summary(groups, n_entities=int((audit["source_id"] == tgt).sum()))
    coverage_full = source_coverage(obs, registry, None)
    coverage_asof = source_coverage(obs, registry, as_of)

    feats = build_feature_table(obs, registry, as_of)
    cols = feature_columns(feats)
    folds, _ = walk_forward(feats, cols, profile=profile)
    gate = evaluate_gate(folds, profile)

    importance = None
    if gate["passed"]:
        train = usable(feats)
        train = train[train["y"].notna()]
        models = fit_models(train, cols)
        imp = source_contributions(models, train[cols].astype(float), column_source_map(cols, registry)).abs().mean()
        importance = {k: float(v) for k, v in imp.items()}

    ablation = run_ablation(obs, registry, as_of, profile)
    verdict = _verdict(gate)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(f"=== Model report as of {C.period_label(as_of)} (target: {tgt}, gate profile: {profile['name']}) ===\n")
    print("1. Signal audit (dispersion ratio ~1 = indistinguishable from sampling noise)")
    print(audit_summary.to_string(index=False), "\n")
    print(f"1b. Duplicate series: {dup_summary['groups']} groups, {dup_summary['entities_collapsed']} entities "
          f"collapsed into canonical members ({dup_summary['series_before']} series -> {dup_summary['series_after']})")
    for g in groups:
        print(f"    canonical {g['canonical']}: members {', '.join(g['members'])} "
              f"(kinds {'/'.join(g['kinds'])}, {g['shared_periods']} shared periods)")
    print()
    print("2. Timeline coverage (all data | known at as_of)")
    cov = coverage_full.merge(coverage_asof[["source_id", "periods", "overlap_periods"]], on="source_id",
                              suffixes=("_all", "_asof"))
    print(cov.to_string(index=False), "\n")
    print("3. Walk-forward folds")
    print(folds.to_string(index=False) if not folds.empty else "(no usable folds)", "\n")
    print("4. Gate verdict")
    print(verdict, "\n")
    print("5. Ablation (leave-one-source-out; no_overlap = not run)")
    print(ablation.to_string(index=False), "\n")
    if importance is not None:
        print("6. Learned source importance (mean |SHAP|)")
        print(json.dumps(importance, indent=2), "\n")
    print("VERDICT:", verdict.splitlines()[0])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": C.period_label(as_of), "target_source": tgt, "gate_profile": profile["name"],
        "disclaimer": registry.get("disclaimer"),
        "audit_summary": records(audit_summary),
        "timeline": {"all_data": records(coverage_full), "known_at_as_of": records(coverage_asof)},
        "folds": records(folds), "gate": gate, "gate_verdict": verdict,
        "ablation": records(ablation), "source_importance": importance,
        "duplicates": {**dup_summary, "groups": groups},
    }
    C.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {C.REPORT_PATH}")


if __name__ == "__main__":
    main()
