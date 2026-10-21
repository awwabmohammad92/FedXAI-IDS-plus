#!/usr/bin/env python3
"""Aggregate runs/*.json -> tables/*.csv with mean, SD, 95% t-CI, Holm-corrected Wilcoxon tests
and TOST equivalence tests (addresses R2.8 and R4 'statistically tying' comment).

Usage: python3 scripts/analyze_v5.py [--margin 0.01]
"""
from __future__ import annotations
import json, glob, sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tables"; OUT.mkdir(exist_ok=True)
MARGIN = float(sys.argv[sys.argv.index("--margin") + 1]) if "--margin" in sys.argv else 0.01
KEYS = ["SPLIT", "PARTITION", "ALPHA", "ATTACK", "N_MAL", "EPS_S", "AGG", "N_TREES", "DEPTH", "CICIDS_FRAC"]
METRICS = ["f1", "auc", "recall", "precision", "fpr", "backdoor_ASR", "uplink_MB_per_client", "rounds", "epsilon"]

def fam(cfg):
    c = dict(cfg)
    if c.get("PARTITION") != "dirichlet": c["ALPHA"] = "-"
    if c.get("ATTACK") == "none": c["N_MAL"] = 0
    return tuple(str(c.get(k, "")) for k in KEYS)

runs = []
for fp in sorted(glob.glob(str(ROOT / "runs_r1" / "*.json"))):
    try: runs.append(json.load(open(fp)))
    except Exception as e: print("skip", fp, e)
groups = defaultdict(list)
for r in runs: groups[fam(r["config"])].append(r)

def ci95(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan"))
    m, s = x.mean(), x.std(ddof=1); h = stats.t.ppf(0.975, len(x) - 1) * s / math.sqrt(len(x))
    return float(m), float(m - h), float(m + h)

def tost_paired(a, b, margin):
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 3 or d.std(ddof=1) == 0: return float("nan")
    se = d.std(ddof=1) / math.sqrt(len(d)); df = len(d) - 1
    p1 = 1 - stats.t.cdf((d.mean() + margin) / se, df)     # H0: diff <= -margin
    p2 = stats.t.cdf((d.mean() - margin) / se, df)         # H0: diff >= +margin
    return float(max(p1, p2))

def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); adj = np.empty_like(ps); m = len(ps); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj

summary_rows, test_rows = [], []
for key, rs in sorted(groups.items()):
    keyd = dict(zip(KEYS, key))
    dsets = sorted({d for r in rs for d in r["datasets"]})
    for ds in dsets:
        per_method = defaultdict(lambda: defaultdict(dict))   # method -> metric -> seed -> value
        for r in rs:
            seed = r["config"]["SEED"]
            for m, v in r["datasets"].get(ds, {}).get("results", {}).items():
                for k in METRICS:
                    if k in v: per_method[m][k][seed] = v[k]
        for m, mets in per_method.items():
            row = {**keyd, "dataset": ds, "method": m, "n_seeds": len(mets.get("f1", {}))}
            for k, sv in mets.items():
                mu, lo, hi = ci95(list(sv.values()))
                row[f"{k}_mean"], row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = mu, lo, hi
            summary_rows.append(row)
        # paired tests of ours vs every other method on F1 (Holm within this table)
        if "ours" in per_method:
            ours = per_method["ours"]["f1"]; comps = []
            for m, mets in per_method.items():
                if m == "ours": continue
                seeds = sorted(set(ours) & set(mets["f1"]))
                if len(seeds) < 3: continue
                a = np.array([ours[s] for s in seeds]); b = np.array([mets["f1"][s] for s in seeds])
                try: p = stats.wilcoxon(a, b, zero_method="pratt").pvalue
                except ValueError: p = 1.0
                comps.append((m, seeds, a, b, p))
            if comps:
                adj = holm([c[4] for c in comps])
                for (m, seeds, a, b, p), pa in zip(comps, adj):
                    test_rows.append({**keyd, "dataset": ds, "comparison": f"ours vs {m}", "n": len(seeds),
                                      "mean_diff_f1": float((a - b).mean()), "wilcoxon_p": float(p),
                                      "holm_p": float(pa), f"tost_p_margin_{MARGIN}": tost_paired(a, b, MARGIN)})

import csv
def write(name, rows):
    if not rows: return
    cols = sorted({k for r in rows for k in r}, key=lambda c: (c not in KEYS + ["dataset", "method", "comparison"], c))
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT/name} ({len(rows)} rows)")
write("summary.csv", summary_rows); write("tests.csv", test_rows)
print(f"{len(runs)} runs, {len(groups)} experiment families")
