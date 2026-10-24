import json, glob, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.py")).read())  # run from results/: python3 ../analysis/make_figs.py
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 150})
OUT = "../figures"; os.makedirs(OUT, exist_ok=True)
DS = ["nslkdd", "cicids", "toniot"]; DSL = {"nslkdd": "NSL-KDD", "cicids": "CIC-IDS-2017", "toniot": "TON_IoT"}
LBL = {"central_lgbm_matched": "LightGBM (central)", "central_xgb": "XGBoost (central)", "central_rf": "RF (central)",
       "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedxgb_ens": "FedXGB-Ens.",
       "fedhist": "FedHist (no DP)", "ours": "FedXAI-IDS+ (ε=8)", "ours_secagg": "FedXAI-IDS+ SecAgg (ε=8)",
       "ours_nodp": "FedXAI-IDS+ (no DP)", "ours_norobust": "w/o robust agg.", "ours_nowarm": "w/o warm start"}

# Fig 1: main comparison with 95% CI
m = fam(s); meths = ["central_lgbm_matched", "central_xgb", "fedavg", "scaffold", "fedxgb_ens", "fedhist", "ours_nodp", "ours_secagg", "ours"]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
for a, ds in zip(ax, DS):
    d = m[m.dataset == ds].set_index("method").loc[meths]
    y = d.f1_mean.values; err = np.vstack([y - d.f1_ci_lo.values, d.f1_ci_hi.values - y])
    a.barh(range(len(meths)), y, xerr=err, color=["#9aa5b1"]*2 + ["#e0b080"]*3 + ["#6c8ebf"]*2 + ["#c0504d"]*2, capsize=2)
    a.set_yticks(range(len(meths))); a.set_yticklabels([LBL[x] for x in meths] if ds == "nslkdd" else [])
    a.set_xlim(0, 1); a.set_title(DSL[ds]); a.set_xlabel("F1 (attack class)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_main_f1.png"); plt.close()

# Fig 2: privacy-utility
fig, ax = plt.subplots(1, 3, figsize=(10, 2.8)); E = [1, 2, 4, 8, 16, np.inf]; xs = np.arange(len(E))
for a, ds in zip(ax, DS):
    for meth, c in [("ours", "#c0504d"), ("ours_nowarm", "#e0b080")]:
        mu, lo, hi = [], [], []
        for e in E:
            d = fam(s, EPS_S=float(e)); d = d[(d.dataset == ds) & (d.method == meth)]
            mu.append(d.f1_mean.iloc[0]); lo.append(d.f1_ci_lo.iloc[0]); hi.append(d.f1_ci_hi.iloc[0])
        a.plot(xs, mu, "o-", color=c, label={"ours":"FedXAI-IDS+","ours_nowarm":"w/o warm start"}[meth]); a.fill_between(xs, lo, hi, color=c, alpha=0.15)
    ref = fam(s); ref = ref[ref.dataset == ds].set_index("method")
    a.axhline(ref.loc["fedhist", "f1_mean"], ls="--", color="#6c8ebf", label="FedHist (no DP)")
    a.axhline(ref.loc["central_lgbm_matched", "f1_mean"], ls=":", color="k", label="LightGBM (central)")
    a.set_xticks(xs); a.set_xticklabels(["1", "2", "4", "8", "16", "∞"]); a.set_xlabel("privacy budget ε (δ=1e-5)")
    a.set_title(DSL[ds]); a.set_ylabel("F1" if ds == "nslkdd" else "")
ax[0].legend(fontsize=7, frameon=False); plt.tight_layout(); plt.savefig(f"{OUT}/fig_privacy_utility.png"); plt.close()

# Fig 3: aggregator x attack heatmap
rows = []
for g in ["cclip", "multikrum", "trimmed", "median"]:
    for at, k in [("none", 0), ("signflip", 2), ("alie", 2), ("backdoor", 2)]:
        d = fam(s, AGG=g, ATTACK=at, N_MAL=k); d = d[d.method == "ours"]
        for _, r in d.iterrows(): rows.append((g, at, r.dataset, r.f1_mean))
X = pd.DataFrame(rows, columns=["agg", "attack", "ds", "f1"])
fig, ax = plt.subplots(1, 3, figsize=(10, 2.6))
AG = ["cclip", "multikrum", "trimmed", "median"]; AGL = ["Centred clip.", "Multi-Krum", "Trimmed mean", "Median"]
AT = ["none", "signflip", "alie", "backdoor"]; ATL = ["No attack", "Sign flip", "ALIE", "Backdoor"]
for a, ds in zip(ax, DS):
    M = X[X.ds == ds].pivot(index="agg", columns="attack", values="f1").loc[AG, AT].values
    im = a.imshow(M, vmin=0.3, vmax=1.0, cmap="RdYlGn")
    for i in range(4):
        for j in range(4): a.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    a.set_xticks(range(4)); a.set_xticklabels(ATL, rotation=30, ha="right"); a.set_yticks(range(4))
    a.set_yticklabels(AGL if ds == "nslkdd" else []); a.set_title(DSL[ds])
plt.tight_layout(); plt.savefig(f"{OUT}/fig_aggregators.png"); plt.close()

# Fig 4: attacks vs malicious clients
fig, ax = plt.subplots(3, 5, figsize=(12, 6.5), sharey=True)
for j, at in enumerate(["labelflip", "signflip", "scale", "alie", "backdoor"]):
    for i, ds in enumerate(DS):
        a = ax[i, j]
        for meth, c in [("fedhist", "#6c8ebf"), ("ours", "#c0504d"), ("ours_norobust", "#e0b080"), ("fedavg", "#999999")]:
            ys = []
            for k in [0, 1, 2]:
                d = fam(s) if k == 0 else fam(s, ATTACK=at, N_MAL=k)
                d = d[(d.dataset == ds) & (d.method == meth)]; ys.append(d.f1_mean.iloc[0] if len(d) else np.nan)
            a.plot([0, 1, 2], ys, "o-", color=c, label=LBL[meth])
        a.set_xticks([0, 1, 2]); a.set_ylim(0, 1)
        if i == 0: a.set_title({"labelflip": "Label flip", "signflip": "Sign flip", "scale": "Scaling", "alie": "ALIE", "backdoor": "Backdoor"}[at])
        if j == 0: a.set_ylabel(f"{DSL[ds]}\nF1")
        if i == 2: a.set_xlabel("malicious clients")
ax[0, 0].legend(fontsize=6, frameon=False); plt.tight_layout(); plt.savefig(f"{OUT}/fig_attacks.png"); plt.close()

# Fig 5: random vs realistic split
fig, ax = plt.subplots(figsize=(6, 2.8)); w = 0.35
for k, (sp, c, lab) in enumerate([("random", "#bbbbbb", "random 70/30 split"), ("default", "#c0504d", "official / temporal split")]):
    d = fam(s, SPLIT=sp); d = d[d.method == "central_lgbm_matched"].set_index("dataset").loc[DS]
    ax.bar(np.arange(3) + k * w, d.f1_mean.values, w, color=c, label=lab)
ax.set_xticks(np.arange(3) + w / 2); ax.set_xticklabels([DSL[d] for d in DS]); ax.set_ylim(0.5, 1.02)
ax.set_ylabel("F1, centralised LightGBM"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_split_optimism.png"); plt.close()

# Fig 6: SHAP top features (mean over 10 seeds)
sh = collections.defaultdict(lambda: collections.defaultdict(float))
for f in glob.glob("runs_r1/A_main_*.json"):
    r = json.load(open(f))
    for ds, v in r["datasets"].items():
        for x in v["info"].get("shap_top15", []): sh[ds][x["feature"]] += x["mean_abs_shap"] / 10
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for a, ds in zip(ax, DS):
    top = sorted(sh[ds].items(), key=lambda kv: -kv[1])[:8][::-1]
    a.barh([k for k, _ in top], [v for _, v in top], color="#c0504d"); a.set_title(DSL[ds]); a.set_xlabel("mean |SHAP| (log-odds)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap.png"); plt.close()

# Fig 7: uplink per stage (default config)
r = json.load(open("runs_r1/A_main_s42.json")); rows = []
for ds in DS:
    i = r["datasets"][ds]["info"]; nf = i["n_features"]
    sketch = nf * 1024 * 4 / 1e6; s1 = 20 * (nf + 1) * 4 / 1e6
    tot = r["datasets"][ds]["results"]["ours"]["uplink_MB_per_client"]; rows.append((ds, sketch, s1, tot - sketch - s1))
U = pd.DataFrame(rows, columns=["ds", "sketch", "stage1", "stage2"])
fig, ax = plt.subplots(figsize=(6, 2.6)); left = np.zeros(3)
for col, c, lab in [("sketch", "#9aa5b1", "DP bin sketch"), ("stage1", "#e0b080", "Stage 1 (warm start)"), ("stage2", "#c0504d", "Stage 2 (histograms)")]:
    ax.barh([DSL[d] for d in U.ds], U[col], left=left, color=c, label=lab); left += U[col].values
ax.set_xlabel("uplink per client (MB), ε=8, T=50, D=5"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_communication.png"); plt.close()
U.to_csv("tables/uplink_by_stage.csv", index=False); print(U.round(3))
print("figures written")
