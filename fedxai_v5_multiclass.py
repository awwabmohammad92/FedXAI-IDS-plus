#!/usr/bin/env python3
"""Multi-class extension of FedXAI-IDS v5 (answers R2.8 / R2.10: multi-class results over
several seeds, consistent class counts). Reuses the v5 data pipeline, DP accountant, binning
and robust aggregation, and adds a softmax histogram GBDT in which the K per-class trees of a
boosting round are grown jointly: one upload per level carries the histograms of all K trees,
so the number of DP releases is the same as in the binary case (R = 1 + T*D).

Per record and level, sum_k g_k^2 <= 2 and sum_k h_k^2 <= 1/4 for every feature, so the L2
sensitivity of one upload is 1.5*sqrt(d).

Usage: SEED=42 DATASETS=nslkdd,toniot TAG=H_mc_s42 python3 fedxai_v5_multiclass.py
"""
from __future__ import annotations
import os, json, time, glob, math, traceback
import numpy as np, pandas as pd
import fedxai_v5 as v5
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb, xgboost as xgb

NSL_FAMILY = {
 **{a: "DoS" for a in ["back","land","neptune","pod","smurf","teardrop","mailbomb","apache2","processtable","udpstorm"]},
 **{a: "Probe" for a in ["ipsweep","nmap","portsweep","satan","mscan","saint"]},
 **{a: "R2L" for a in ["ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster",
                       "sendmail","named","snmpgetattack","snmpguess","xlock","xsnoop","worm","httptunnel"]},
 **{a: "U2R" for a in ["buffer_overflow","loadmodule","perl","rootkit","sqlattack","ps","xterm"]},
 "normal": "normal"}
CIC_FAMILY = {"BENIGN": "Benign", "DDoS": "DDoS", "PortScan": "PortScan", "Bot": "Bot", "Infiltration": "Infiltration",
              "Heartbleed": "DoS", "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS",
              "DoS Slowhttptest": "DoS", "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce"}

def load_mc(name):
    """Returns tr, ytr(str), te, yte(str), groups, split. Same preprocessing as v5."""
    if name == "nslkdd":
        a, b = v5.DATA_ROOT / "nslkdd" / "KDDTrain.txt", v5.DATA_ROOT / "nslkdd" / "KDDTest.txt"
        tr = pd.read_csv(a, header=None, names=v5.NSL_COLS).drop(columns=["difficulty"])
        te = pd.read_csv(b, header=None, names=v5.NSL_COLS).drop(columns=["difficulty"])
        g = tr["service"].astype(str).values
        tr, te = v5._encode_cats(tr, te, ["protocol_type", "service", "flag"])
        ytr = tr.pop("label").map(lambda s: NSL_FAMILY.get(s, "other")).values
        yte = te.pop("label").map(lambda s: NSL_FAMILY.get(s, "other")).values
        return tr, ytr, te, yte, g, "official"
    if name == "toniot":
        f = sorted(glob.glob(str(v5.DATA_ROOT / "toniot" / "*.csv")))[0]
        df = pd.read_csv(f, low_memory=False, encoding="utf-8-sig")
        typ = df["type"].astype(str).values
        m = np.zeros(len(df), bool)
        for t in np.unique(typ):
            ix = np.where(typ == t)[0]; m[ix[: int(0.7 * len(ix))]] = True
        cats = [c for c in ["proto", "service", "conn_state"] if c in df.columns]
        X = df.drop(columns=[c for c in ["ts", "src_ip", "dst_ip", "src_port", "dst_port", "label", "type"] if c in df.columns])
        X = X[[c for c in X.columns if c in cats or pd.api.types.is_numeric_dtype(X[c])]]
        tr, te = X[m].reset_index(drop=True), X[~m].reset_index(drop=True)
        tr, te = v5._encode_cats(tr, te, cats)
        return tr.astype(float), typ[m], te.astype(float), typ[~m], df["dst_ip"].astype(str).values[m], "type-stratified order"
    if name == "cicids":
        parts = []
        for i, fp in enumerate(sorted(glob.glob(str(v5.DATA_ROOT / "cicids" / "*.csv")))):
            d = pd.read_csv(fp, low_memory=False); d.columns = [c.strip() for c in d.columns]
            d = d.replace([np.inf, -np.inf], np.nan).dropna()
            keep = np.sort(v5.rng_global.choice(len(d), int(v5.CICIDS_FRAC * len(d)), replace=False)); d = d.iloc[keep]
            d["_f"] = i; d["_p"] = np.arange(len(d)) / max(1, len(d) - 1); parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        lab = df.pop("Label").astype(str).str.strip()
        y = np.array([CIC_FAMILY.get(l, "WebAttack" if l.startswith("Web Attack") else l) for l in lab])
        fid, pos = df.pop("_f").values, df.pop("_p").values
        X = df.select_dtypes(include=[np.number]).astype(float); m = pos < 0.7
        return (X[m].reset_index(drop=True), y[m], X[~m].reset_index(drop=True), y[~m], fid[m].astype(str), "temporal")
    return None

def softmax(F):
    F = F - F.max(1, keepdims=True); e = np.exp(F); return e / e.sum(1, keepdims=True)

def fed_softmax_gbdt(Xc, yc, K, z, robust, nb, nf):
    sens = 1.5 * math.sqrt(nf); nC = len(yc); n_i = np.array([len(y) for y in yc], float)
    sd = z * sens; min_hess = max(v5.MIN_HESS, sd * math.sqrt(nC) * math.sqrt(nb))
    Xoff = [X.astype(np.int64) + np.arange(nf)[None, :] * nb for X in Xc]
    Fc = [np.zeros((len(y), K)) for y in yc]; Y = [np.eye(K)[y] for y in yc]
    ensembles, up = [], 0
    for t in range(v5.N_TREES):
        trees = [v5.Tree() for _ in range(K)]
        for tr in trees: tr.add(1.0)
        node = [[np.zeros(len(y), np.int64) for y in yc] for _ in range(K)]
        level = [[0] for _ in range(K)]
        GH = []
        for i in range(nC):
            P = softmax(Fc[i]); GH.append((P - Y[i], P * (1 - P)))
        for lvl in range(v5.DEPTH):
            Ks = [len(level[k]) for k in range(K)]
            if sum(Ks) == 0: break
            vecs = []
            for i in range(nC):
                parts_G, parts_H = [], []
                for k in range(K):
                    if Ks[k] == 0: continue
                    act = node[k][i] >= 0; L = Ks[k] * nf * nb
                    if act.any():
                        idx = (node[k][i][act][:, None] * (nf * nb) + Xoff[i][act]).ravel()
                        G = np.bincount(idx, weights=np.repeat(GH[i][0][act, k], nf), minlength=L)
                        H = np.bincount(idx, weights=np.repeat(GH[i][1][act, k], nf), minlength=L)
                    else: G = np.zeros(L); H = np.zeros(L)
                    parts_G.append(G); parts_H.append(H)
                G = np.concatenate(parts_G); H = np.concatenate(parts_H)
                if z > 0:
                    G = G + v5.rng_global.normal(0, sd, G.shape); H = H + v5.rng_global.normal(0, sd, H.shape)
                vecs.append(np.concatenate([G, H]) / n_i[i])
            up += len(vecs[0]) * 4
            if robust: agg, _ = v5.aggregate(vecs, n_i)
            else: agg = (np.stack(vecs) * n_i[:, None]).sum(0) / n_i.sum()
            agg = agg * n_i.sum(); half = len(agg) // 2; Gall, Hall = agg[:half], np.clip(agg[half:], 0, None)
            off = 0; remaps = []
            for k in range(K):
                if Ks[k] == 0: remaps.append({}); continue
                L = Ks[k] * nf * nb
                G = Gall[off:off + L].reshape(Ks[k], nf, nb); H = Hall[off:off + L].reshape(Ks[k], nf, nb); off += L
                GL, HL = np.cumsum(G, 2)[:, :, :-1], np.cumsum(H, 2)[:, :, :-1]
                Gt, Ht = G.sum(2, keepdims=True), H.sum(2, keepdims=True); GR, HR = Gt - GL, Ht - HL
                lam = v5.LAMBDA
                gain = GL**2 / (HL + lam) + GR**2 / (HR + lam) - Gt**2 / (Ht + lam)
                gain[(HL < min_hess) | (HR < min_hess)] = -np.inf
                tree, nxt, remap = trees[k], [], {}
                for j, nd in enumerate(level[k]):
                    gj = gain[j]; best = np.unravel_index(np.argmax(gj), gj.shape)
                    Gk, Hk = float(Gt[j, best[0], 0]), float(Ht[j, best[0], 0])
                    lr = v5.LR * (K - 1) / K
                    if not np.isfinite(gj[best]) or gj[best] <= 0:
                        tree.val[nd] = -lr * Gk / (Hk + lam); remap[j] = None; continue
                    f, b = int(best[0]), int(best[1]); tree.feat[nd], tree.thr[nd] = f, b
                    l = tree.add(HL[j, f, b]); r = tree.add(HR[j, f, b]); tree.left[nd], tree.right[nd] = l, r
                    if lvl == v5.DEPTH - 1:
                        tree.val[l] = -lr * float(GL[j, f, b]) / (float(HL[j, f, b]) + lam)
                        tree.val[r] = -lr * float(GR[j, f, b]) / (float(HR[j, f, b]) + lam); remap[j] = None
                    else:
                        remap[j] = (f, b, len(nxt), len(nxt) + 1); nxt += [l, r]
                level[k] = nxt; remaps.append(remap)
            for k in range(K):
                for i in range(nC):
                    nc = node[k][i]; new = np.full_like(nc, -1)
                    for j, info in remaps[k].items():
                        if info is None: continue
                        f, b, li, ri = info; m = nc == j
                        new[m] = np.where(Xc[i][m, f] <= b, li, ri)
                    node[k][i] = new
        ensembles.append(trees)
        for i in range(nC):
            for k in range(K): Fc[i][:, k] += trees[k].predict(Xc[i])
    return ensembles, up

def predict_mc(ens, Xb, K):
    F = np.zeros((len(Xb), K))
    for trees in ens:
        for k in range(K): F[:, k] += trees[k].predict(Xb)
    return F.argmax(1)

def run(name):
    t0 = time.time(); L = load_mc(name)
    if L is None: return None
    tr, ytr_s, te, yte_s, g, split = L
    classes = sorted(set(ytr_s) | set(yte_s)); cid = {c: i for i, c in enumerate(classes)}; K = len(classes)
    ytr, yte = np.array([cid[c] for c in ytr_s]), np.array([cid[c] for c in yte_s])
    tr, ytr, te, yte, g, dstats = v5.dedup(tr, ytr, te, yte, g)
    Atr, Ate = np.arcsinh(tr.values), np.arcsinh(te.values); nf = Atr.shape[1]
    clients = [c for c in v5.make_clients(ytr, g, v5.N_CLIENTS) if len(c)]
    R = 1 + v5.N_TREES * v5.DEPTH; z = v5.z_for_epsilon(v5.EPS, R, v5.DELTA)
    edges, _ = v5.dp_bin_edges([Atr[c] for c in clients], z); nb = max(len(e) for e in edges) + 1
    Xc = [v5.apply_bins(Atr[c], edges) for c in clients]; yc = [ytr[c] for c in clients]; Xte = v5.apply_bins(Ate, edges)
    counts = {"train": {c: int((ytr == cid[c]).sum()) for c in classes}, "test": {c: int((yte == cid[c]).sum()) for c in classes}}
    print(f"\n=== {name} K={K} classes={classes} train={len(ytr)} test={len(yte)} z={z:.2f}", flush=True)
    res = {}
    def rec(m, pred, t):
        res[m] = {"macro_f1": float(f1_score(yte, pred, average="macro", labels=range(K), zero_division=0)),
                  "accuracy": float(accuracy_score(yte, pred)),
                  "per_class_f1": dict(zip(classes, map(float, f1_score(yte, pred, average=None, labels=range(K), zero_division=0)))),
                  "train_seconds": time.time() - t}
        print(f"  {m:22s} macroF1={res[m]['macro_f1']:.4f} acc={res[m]['accuracy']:.4f}", flush=True)
    t = time.time(); rec("central_rf", RandomForestClassifier(200, n_jobs=-1, random_state=v5.SEED).fit(Atr, ytr).predict(Ate), t)
    t = time.time(); m = lgb.LGBMClassifier(n_estimators=v5.N_TREES, max_depth=v5.DEPTH, num_leaves=2**v5.DEPTH, learning_rate=v5.LR,
                                         max_bin=v5.NBINS, verbosity=-1, random_state=v5.SEED, n_jobs=-1).fit(Atr, ytr)
    rec("central_lgbm_matched", m.predict(Ate), t)
    t = time.time(); m = xgb.train({"objective": "multi:softprob", "num_class": K, "max_depth": v5.DEPTH, "eta": v5.LR,
                                    "max_bin": v5.NBINS, "tree_method": "hist", "verbosity": 0, "seed": v5.SEED},
                                   xgb.DMatrix(Atr, label=ytr), v5.N_TREES)
    rec("central_xgb", m.predict(xgb.DMatrix(Ate)).argmax(1), t)
    for mname, zz, robust in [("fedhist", 0.0, False), ("ours", z, True), ("ours_nodp", 0.0, True)]:
        t = time.time(); ens, up = fed_softmax_gbdt(Xc, yc, K, zz, robust, nb, nf)
        rec(mname, predict_mc(ens, Xte, K), t); res[mname]["uplink_MB_per_client"] = up / 1e6
    return {"results": res, "info": {"classes": classes, "class_counts": counts, "split": split, "data": dstats,
            "n_clients": len(clients), "dp": {"eps": v5.EPS, "delta": v5.DELTA, "z": z, "releases": R},
            "wall_seconds": time.time() - t0}}

if __name__ == "__main__":
    out = {"config": {k: getattr(v5, k) for k in ["SEED", "EPS_S", "AGG", "N_TREES", "DEPTH", "PARTITION", "CICIDS_FRAC"]}
           | {"TAG": v5.TAG, "MULTICLASS": True}, "datasets": {}}
    for ds in v5.DATASETS:
        try:
            r = run(ds)
            if r: out["datasets"][ds] = r
        except Exception as e:
            print(f"[error {ds}] {e}"); traceback.print_exc()
    fp = v5.RUNS / f"{v5.TAG}.json"
    with open(fp, "w") as f: json.dump(out, f, indent=1, default=float)
    print(f"DONE -> {fp}")
