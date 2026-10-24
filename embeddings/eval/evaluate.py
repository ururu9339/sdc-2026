import argparse
import csv
import hashlib
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from embeddings import db, embedder, versioning

HERE = Path(__file__).parent
GOLD_MENTIONS = HERE / "gold_mentions.csv"
GOLD_PAIRS = HERE / "gold_job_pairs.csv"
TOP_K = 5
HIGH_PRECISION = 0.95
BAND_PRECISION = 0.50
COLD_START_PRECISION = 0.90 

def gold_sha():
    h = hashlib.sha256()
    for p in (GOLD_MENTIONS, GOLD_PAIRS):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]

def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

@contextmanager
def using_model(name):
    old = embedder.MODEL
    embedder.MODEL = name
    try:
        yield
    finally:
        embedder.MODEL = old

def embedding_ranker(gold, taxonomy):
    out = [None] * len(gold)
    for etype in ("skill", "job"):
        ents = [r for r in taxonomy if r["type"] == etype]
        D = embedder.embed_documents([embedder.entity_text(r) for r in ents])
        idx = [i for i, g in enumerate(gold) if g["entity_type"] == etype]
        Q = embedder.embed_queries([gold[i]["mention"] for i in idx])
        S = Q @ D.T
        for row, i in zip(S, idx):
            top = np.argsort(-row)[:TOP_K]
            out[i] = [(ents[j]["id"], float(row[j])) for j in top]
    return out

def _labels(taxonomy, etype):
    labels = []
    for r in taxonomy:
        if r["type"] == etype:
            labels += [(r["name"], r["id"])] + [(a, r["id"]) for a in r["alt_labels"].split("|") if a]
    return labels


def exact_ranker(gold, taxonomy, lowercase):
    key = _norm if lowercase else (lambda s: s)
    index = {}
    for etype in ("skill", "job"):
        for label, eid in _labels(taxonomy, etype):
            index.setdefault((etype, key(label)), eid)
    return [[(index[(g["entity_type"], key(g["mention"]))], 1.0)]
            if (g["entity_type"], key(g["mention"])) in index else [] for g in gold]

def fuzzy_ranker(gold, taxonomy):
    choices = {t: _labels(taxonomy, t) for t in ("skill", "job")}
    out = []
    for g in gold:
        labels = choices[g["entity_type"]]
        hits = process.extract(g["mention"], [l for l, _ in labels], scorer=fuzz.WRatio,
                               processor=_norm, limit=40)
        best = {}
        for _, score, i in hits:
            eid = labels[i][1]
            best[eid] = max(best.get(eid, 0.0), score / 100)
        out.append(sorted(best.items(), key=lambda kv: -kv[1])[:TOP_K])
    return out

def score(gold, ranked, etype=None):
    rows = [(g, r) for g, r in zip(gold, ranked) if etype is None or g["entity_type"] == etype]
    pos = [(g, r) for g, r in rows if g["gold_id"] != "NONE"]
    ranks = []
    for g, r in pos:
        ids = [e for e, _ in r]
        ranks.append(ids.index(g["gold_id"]) + 1 if g["gold_id"] in ids else None)
    m = {
        "n": len(pos), "negatives": len(rows) - len(pos),
        "top1": sum(k == 1 for k in ranks) / len(pos),
        "top5": sum(k is not None and k <= 5 for k in ranks) / len(pos),
        "mrr": sum(1 / k for k in ranks if k) / len(pos),
    }
    sweep = []
    for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        acc = [(g, r) for g, r in rows if r and r[0][1] >= t]
        correct = sum(r[0][0] == g["gold_id"] for g, r in acc)
        neg_rejected = sum(1 for g, r in rows if g["gold_id"] == "NONE" and not (r and r[0][1] >= t))
        sweep.append({"t": float(t), "accepted": len(acc), "correct": correct, "precision": correct / len(acc) if acc else 1.0,
                      "recall": correct / len(pos), "neg_rejected": neg_rejected / max(len(rows) - len(pos), 1)})
    return m, sweep

def pick(sweep, precision, min_accepted=10):
    for s in sweep:
        if s["precision"] >= precision and s["accepted"] >= min_accepted:
            return s
    return None

def pick_low(sweep, high, precision=BAND_PRECISION):
    h = next(s for s in sweep if s["t"] == high["t"])
    for s in sweep:
        if s["t"] >= high["t"]:
            return high
        n = s["accepted"] - h["accepted"]
        if n and (s["correct"] - h["correct"]) / n >= precision:
            return {**s, "band_precision": (s["correct"] - h["correct"]) / n, "band_n": n}
    return high

def auc(scores, labels):
    p = [s for s, l in zip(scores, labels) if l]
    n = [s for s, l in zip(scores, labels) if not l]
    return sum((a > b) + 0.5 * (a == b) for a in p for b in n) / (len(p) * len(n))

def pair_scores(pairs, method):
    if method == "fuzzy":
        return [fuzz.WRatio(p["title_a"], p["title_b"], processor=_norm) / 100 for p in pairs]
    A = embedder.embed_queries([p["title_a"] for p in pairs])
    B = embedder.embed_queries([p["title_b"] for p in pairs])
    return list((A * B).sum(axis=1))

def evaluate(model=None):
    gold, pairs, taxonomy = _read(GOLD_MENTIONS), _read(GOLD_PAIRS), embedder.load_taxonomy()
    labels = [int(p["similar"]) for p in pairs]
    alias = {(r["type"], _norm(l)) for r in taxonomy for l in [r["name"], *r["alt_labels"].split("|")] if l}
    unseen = [i for i, g in enumerate(gold) if (g["entity_type"], _norm(g["mention"])) not in alias]

    methods = {"exact": lambda: exact_ranker(gold, taxonomy, False),
               "lowercase": lambda: exact_ranker(gold, taxonomy, True),
               "rapidfuzz": lambda: fuzzy_ranker(gold, taxonomy)}
    models = [embedder.MODEL] + ([model] if model else [])
    results = {}
    for name, fn in methods.items():
        results[name] = {"ranked": fn(), "pairs": pair_scores(pairs, "fuzzy") if name == "rapidfuzz" else None}
    for m in models:
        with using_model(m):
            results[m] = {"ranked": embedding_ranker(gold, taxonomy), "pairs": pair_scores(pairs, "embedding")}

    report = {}
    for name, r in results.items():
        overall, sweep = score(gold, r["ranked"])
        ug = [gold[i] for i in unseen]
        ur = [r["ranked"][i] for i in unseen]
        report[name] = {
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in overall.items()},
            "top1_unseen": round(score(ug, ur)[0]["top1"], 4),
            "top1_skill": round(score(gold, r["ranked"], "skill")[0]["top1"], 4),
            "top1_job": round(score(gold, r["ranked"], "job")[0]["top1"], 4),
            "pair_auc": round(auc(r["pairs"], labels), 4) if r["pairs"] is not None else None,
            "_sweep": sweep, "_job_sweep": score(gold, r["ranked"], "job")[1],
        }
    return report, len(unseen), len(gold)

def thresholds_for(rep):
    hi = pick(rep["_sweep"], HIGH_PRECISION)
    lo = pick_low(rep["_sweep"], hi) if hi else None
    cs = pick(rep["_job_sweep"], COLD_START_PRECISION, min_accepted=5)
    return ({"high": hi["t"] if hi else None, "low": lo["t"] if lo else None,
             "cold_start_min": cs["t"] if cs else None}, hi, lo, cs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", help="another Ollama embedding model to score, e.g. qwen3-embedding:0.6b")
    p.add_argument("--record", action="store_true", help="log the current model as active if it passes the gate")
    p.add_argument("--force", action="store_true", help="record even if the gate fails or the gold set changed")
    args = p.parse_args()

    report, n_unseen, n_gold = evaluate(args.compare)
    print(f"Gold set {gold_sha()}: {n_gold} mentions ({n_unseen} not verbatim taxonomy labels)\n")
    print(f"{'method':24s} {'top1':>6s} {'top5':>6s} {'MRR':>6s} {'unseen':>7s} {'skill':>6s} {'job':>6s} {'pairAUC':>8s}")
    for name, r in report.items():
        auc_s = f"{r['pair_auc']:.3f}" if r["pair_auc"] is not None else "   -"
        print(f"{name:24s} {r['top1']:6.3f} {r['top5']:6.3f} {r['mrr']:6.3f} {r['top1_unseen']:7.3f} "
              f"{r['top1_skill']:6.3f} {r['top1_job']:6.3f} {auc_s:>8s}")

    for name in [embedder.MODEL] + ([args.compare] if args.compare else []):
        th, hi, lo, cs = thresholds_for(report[name])
        print(f"\n{name} thresholds")
        for label, s, target in (("high", hi, HIGH_PRECISION), ("low", lo, BAND_PRECISION),
                                 ("cold_start_min (jobs)", cs, COLD_START_PRECISION)):
            if s:
                extra = f" band [{s['t']:.2f}, high) precision {s['band_precision']:.2f} on {s['band_n']}" if "band_n" in s else ""
                print(f"  {label:22s} {s['t']:.2f}  precision {s['precision']:.3f} (target {target}){extra} "
                      f"recall {s['recall']:.3f}  negatives rejected {s['neg_rejected']:.0%}")
            else:
                print(f"  {label:22s} none reaches precision {target}")

    if not args.record:
        return
    r = report[embedder.MODEL]
    metrics = {k: r[k] for k in ("top1", "top5", "mrr", "top1_unseen", "top1_skill", "top1_job", "pair_auc")}
    current = versioning.active()
    problems = versioning.gate(metrics, current)
    if current and current.get("gold_sha") != gold_sha():
        problems.append(f"gold set changed ({current.get('gold_sha')} -> {gold_sha()}); scores are not comparable")
    if problems and not args.force:
        print("\nGate failed, not recorded:\n  " + "\n  ".join(problems))
        sys.exit(1)
    th = thresholds_for(r)[0]
    versioning.record({
        "version_tag": embedder.version_tag(), "model": embedder.MODEL, "dim": embedder.DIM,
        "task": embedder.TASK, "model_digest": db.ollama_model_digest(), "gold_sha": gold_sha(),
        "metrics": metrics, "thresholds": th,
        "baselines": {k: {"top1": report[k]["top1"], "mrr": report[k]["mrr"]} for k in ("exact", "lowercase", "rapidfuzz")},
        **({"compared": {args.compare: {k: report[args.compare][k] for k in ("top1", "mrr", "pair_auc")}}}
           if args.compare else {}),
    })
    print(f"\nRecorded {embedder.version_tag()} as active in {versioning.LOG}")

if __name__ == "__main__":
    main()
