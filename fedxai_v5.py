#!/usr/bin/env python3
"""
FedXAI-IDS v5 -- revision driver for Array manuscript ARRAY-D-26-04820 (major revision).

What changed relative to v4, mapped to reviewer comments (R2 = Reviewer 2, R4 = Reviewer 4):

  R2.1 / R4.iii  Stage 2 is now a genuinely federated histogram GBDT. Clients upload only
                 (noised) per-bin gradient/hessian sums; no raw records, no per-class
                 feature means and no "training contributions" leave a client.
  R2.3 / R4.iii  Record-level (epsilon, delta)-DP for the *whole* pipeline: DP quantile
                 sketch for bin edges + DP-FedSGD warm-up + DP histograms. One RDP
                 accountant (rdp_epsilon) covers every release.
  R2.4 / R4.i-ii Robust aggregation is standard Multi-Krum (Blanchard et al., 2017) applied
                 to per-client contributions (also: coordinate-wise trimmed mean / median).
                 Attacks: label flip, sign flip, scaling, ALIE (statistics-preserving,
                 Baruch et al., 2019) and a feature-trigger backdoor (attack success rate).
  R2.5           ours_nowarm ablation (no FedSGD warm-up).
  R2.6 / R4.iv   Established federated tree baseline: horizontal histogram GBDT in the style
                 of FedTree (Li et al., 2023), plus capacity-matched centralised LightGBM.
                 SCAFFOLD now applies control-variate corrections inside local steps.
  R2.7 / R4.v-vi Duplicate removal, train-only preprocessing, official / temporal splits,
                 natural (non-Dirichlet) client partitions, leaky identifiers dropped.
  R2.9 / R4.vii  Byte-exact uplink accounting for every stage.
  R2.10          TreeSHAP on the federated ensemble, written to the run JSON.

Usage (env vars, all optional):
  SEED=42 DATASETS=nslkdd,cicids,toniot SPLIT=default PARTITION=natural N_CLIENTS=10
  ATTACK=none|labelflip|signflip|scale|alie|backdoor N_MAL=0 KRUM_F=2 AGG=cclip DP_MODE=local
  EPS=8 DELTA=1e-5 N_TREES=50 DEPTH=5 LR=0.3 LAMBDA=1.0 NBINS=64 T1=20
  METHODS=all CICIDS_FRAC=0.3 SHAP=1 TAG=run DATA_ROOT=./data
"""
from __future__ import annotations
import os, json, time, glob, math, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs_r1"; RUNS.mkdir(exist_ok=True)   # v5.1 outputs (earlier runs/ are obsolete)

# ----------------------------------------------------------------------------- config
E = os.environ.get
SEED       = int(E("SEED", 42))
DATASETS   = [d.strip().lower() for d in E("DATASETS", "nslkdd,cicids,toniot").split(",") if d.strip()]
SPLIT      = E("SPLIT", "default")          # default | random | official | temporal | scenario
PARTITION  = E("PARTITION", "natural")      # natural | iid | dirichlet
ALPHA      = float(E("ALPHA", 0.1))
N_CLIENTS  = int(E("N_CLIENTS", 10))
ATTACK     = E("ATTACK", "none")
N_MAL      = int(E("N_MAL", 0))
KRUM_F     = int(E("KRUM_F", 2))
AGG        = E("AGG", "cclip")              # cclip | multikrum | trimmed | median | mean
DP_MODE    = E("DP_MODE", "local")          # local: each client adds full noise; secagg: distributed noise
EPS_S      = E("EPS", "8")
EPS        = float("inf") if EPS_S.lower() in ("inf", "none") else float(EPS_S)
DELTA      = float(E("DELTA", 1e-5))
N_TREES    = int(E("N_TREES", 50))
DEPTH      = int(E("DEPTH", 5))
LR         = float(E("LR", 0.3))
LAMBDA     = float(E("LAMBDA", 1.0))
NBINS      = int(E("NBINS", 64))
MIN_HESS   = float(E("MIN_HESS", 5.0))
T1         = int(E("T1", 20))
LR1        = float(E("LR1", 2.0))
FINE       = 1024                            # resolution of the DP quantile sketch
LIM        = 25.0                            # asinh range covered by the sketch
METHODS    = E("METHODS", "all")
CICIDS_FRAC= float(E("CICIDS_FRAC", 0.3))
DO_SHAP    = E("SHAP", "1") == "1"
TAG        = E("TAG", "run")
DATA_ROOT  = Path(E("DATA_ROOT", str(ROOT / "data")))
ALIE_Z     = float(E("ALIE_Z", 1.0))

ALL_METHODS = ["central_logreg", "central_rf", "central_xgb", "central_lgbm_matched",
               "central_lgbm_default", "fedavg", "fedprox", "scaffold", "fedxgb_ens",
               "fedhist", "ours", "ours_nowarm", "ours_nodp", "ours_norobust", "ours_secagg"]
RUN_METHODS = ALL_METHODS if METHODS == "all" else [m.strip() for m in METHODS.split(",")]

rng_global = np.random.default_rng(SEED)

def log(msg): print(msg, flush=True)

# ----------------------------------------------------------------------------- DP accountant
def rdp_epsilon(z: float, releases: int, delta: float) -> float:
    """(eps, delta)-DP of `releases` adaptive Gaussian mechanisms, each with noise std z*sensitivity.
    RDP of one Gaussian mechanism: alpha / (2 z^2) (Mironov 2017). Composition adds RDP.
    Conversion: Balle et al. (2020), Prop. 12 / Canonne et al. (2020)."""
    if z <= 0 or not math.isfinite(z): return float("inf")
    a = np.concatenate([np.linspace(1.01, 10, 400), np.arange(11, 2000, dtype=float)])
    rdp = releases * a / (2.0 * z * z)
    eps = rdp + np.log((a - 1) / a) - (np.log(delta) + np.log(a)) / (a - 1)
    return float(eps.min())

def z_for_epsilon(eps: float, releases: int, delta: float) -> float:
    if not math.isfinite(eps): return 0.0
    lo, hi = 1e-3, 1e5
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if rdp_epsilon(mid, releases, delta) > eps: lo = mid
        else: hi = mid
    return hi

# ----------------------------------------------------------------------------- loaders
NSL_COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"]

def _encode_cats(tr: pd.DataFrame, te: pd.DataFrame, cols):
    """Category codes fitted on the training split only; unseen test categories -> -1."""
    for c in cols:
        cats = pd.Index(tr[c].astype(str).unique())
        tr[c] = cats.get_indexer(tr[c].astype(str)).astype(float)
        te[c] = cats.get_indexer(te[c].astype(str)).astype(float)
    return tr, te

def load_nslkdd(split):
    a, b = DATA_ROOT / "nslkdd" / "KDDTrain.txt", DATA_ROOT / "nslkdd" / "KDDTest.txt"
    if not (a.exists() and b.exists()): return None
    tr = pd.read_csv(a, header=None, names=NSL_COLS).drop(columns=["difficulty"])
    te = pd.read_csv(b, header=None, names=NSL_COLS).drop(columns=["difficulty"])
    if split in ("default", "official"):
        split_used = "official (KDDTrain+ / KDDTest+, test contains novel attack types)"
    else:
        df = pd.concat([tr, te], ignore_index=True)
        idx = rng_global.permutation(len(df)); n = int(0.7 * len(df))
        tr, te = df.iloc[idx[:n]].reset_index(drop=True), df.iloc[idx[n:]].reset_index(drop=True)
        split_used = "random 70/30"
    group_tr = tr["service"].astype(str).values          # natural partition: by service
    tr, te = _encode_cats(tr, te, ["protocol_type", "service", "flag"])
    ytr = (tr.pop("label") != "normal").astype(int).values
    yte = (te.pop("label") != "normal").astype(int).values
    return tr, ytr, te, yte, group_tr, split_used

def load_cicids(split):
    files = sorted(glob.glob(str(DATA_ROOT / "cicids" / "*.csv")))
    if not files: return None
    order = ["Monday", "Tuesday", "Wednesday", "Thursday-WorkingHours-Morning",
             "Thursday-WorkingHours-Afternoon", "Friday-WorkingHours-Morning",
             "Friday-WorkingHours-Afternoon-PortScan", "Friday-WorkingHours-Afternoon-DDos"]
    def rank(f):
        b = os.path.basename(f)
        for i, o in enumerate(order):
            if b.startswith(o): return i
        return 99
    files = sorted(files, key=rank)
    parts = []
    for i, f in enumerate(files):
        d = pd.read_csv(f, low_memory=False); d.columns = [c.strip() for c in d.columns]
        if "Timestamp" in d.columns:                     # TrafficLabelling variant
            d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce", dayfirst=True)
            d = d.sort_values("Timestamp", kind="stable")
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        if CICIDS_FRAC < 1.0:                            # order-preserving subsample
            keep = np.sort(rng_global.choice(len(d), int(CICIDS_FRAC * len(d)), replace=False))
            d = d.iloc[keep]
        d["_file"] = i; d["_pos"] = np.arange(len(d)) / max(1, len(d) - 1)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    drop = [c for c in ["Flow ID", "Source IP", "Destination IP", "Source Port", "Timestamp"] if c in df.columns]
    df = df.drop(columns=drop)
    y = (df.pop("Label").astype(str).str.strip().str.upper() != "BENIGN").astype(int).values
    fid, pos = df.pop("_file").values, df.pop("_pos").values
    df = df.select_dtypes(include=[np.number]).astype(float)
    if split in ("default", "temporal"):
        m = pos < 0.7; split_used = "temporal (first 70% of each capture file -> train, rest -> test)"
    elif split == "scenario":
        m = fid < 5; split_used = "scenario (Mon-Thu -> train, Fri captures -> test)"
    else:
        m = rng_global.random(len(df)) < 0.7; split_used = "random 70/30"
    tr, te = df[m].reset_index(drop=True), df[~m].reset_index(drop=True)
    return tr, y[m], te, y[~m], fid[m].astype(str), split_used

def load_toniot(split):
    files = sorted(glob.glob(str(DATA_ROOT / "toniot" / "*.csv")))
    if not files: return None
    df = pd.read_csv(files[0], low_memory=False, encoding="utf-8-sig")
    y = (df["label"].astype(int).values if "label" in df.columns
         else (df["type"].astype(str) != "normal").astype(int).values)
    ts = df["ts"].values if "ts" in df.columns else np.arange(len(df))
    group = df["dst_ip"].astype(str).values if "dst_ip" in df.columns else np.zeros(len(df)).astype(str)
    cats = [c for c in ["proto", "service", "conn_state"] if c in df.columns]
    leaky = ["ts", "src_ip", "dst_ip", "src_port", "dst_port", "label", "type"]
    X = df.drop(columns=[c for c in leaky if c in df.columns])
    X = X[[c for c in X.columns if c in cats or pd.api.types.is_numeric_dtype(X[c])]]
    if split in ("default", "temporal"):
        # A global time cut puts only attack traffic after the cut, so we split
        # chronologically *within* each traffic type (normal and each attack class).
        typ = df["type"].astype(str).values if "type" in df.columns else y.astype(str)
        m = np.zeros(len(df), dtype=bool)
        for t in np.unique(typ):
            ix = np.where(typ == t)[0]
            m[ix[ts[ix] <= np.quantile(ts[ix], 0.7)]] = True
        split_used = ("type-stratified temporal (earliest 70% of each traffic type by ts -> train)"
                      if "ts" in df.columns else
                      "type-stratified order split (no timestamps in this file: first 70% of rows of each traffic type -> train)")
    else:
        m = rng_global.random(len(df)) < 0.7; split_used = "random 70/30"
    tr, te = X[m].reset_index(drop=True), X[~m].reset_index(drop=True)
    tr, te = _encode_cats(tr, te, cats)
    return tr.astype(float), y[m], te.astype(float), y[~m], group[m], split_used

LOADERS = {"nslkdd": load_nslkdd, "cicids": load_cicids, "toniot": load_toniot}

def dedup(tr, ytr, te, yte, gtr):
    """Remove exact duplicate (features+label) rows inside each split and drop test rows whose
    feature vector also occurs in train (cross-split leakage)."""
    stats = {"train_rows_raw": int(len(tr)), "test_rows_raw": int(len(te))}
    htr = pd.util.hash_pandas_object(tr, index=False).values
    hte = pd.util.hash_pandas_object(te, index=False).values
    k_tr = ~pd.Series(list(zip(htr, ytr))).duplicated().values
    tr, ytr, gtr, htr = tr[k_tr].reset_index(drop=True), ytr[k_tr], gtr[k_tr], htr[k_tr]
    k_te = ~pd.Series(list(zip(hte, yte))).duplicated().values
    te, yte, hte = te[k_te].reset_index(drop=True), yte[k_te], hte[k_te]
    k_x = ~np.isin(hte, htr)
    te, yte = te[k_x].reset_index(drop=True), yte[k_x]
    stats.update(train_rows=int(len(tr)), test_rows=int(len(te)),
                 cross_split_dups_removed=int((~k_x).sum()),
                 train_pos_rate=float(ytr.mean()), test_pos_rate=float(yte.mean()))
    return tr, ytr, te, yte, gtr, stats

# ----------------------------------------------------------------------------- partitions
def make_clients(ytr, gtr, n_clients):
    if PARTITION == "iid":
        return [np.sort(c) for c in np.array_split(rng_global.permutation(len(ytr)), n_clients)]
    if PARTITION == "dirichlet":
        cl = [[] for _ in range(n_clients)]
        for c in np.unique(ytr):
            ix = rng_global.permutation(np.where(ytr == c)[0])
            cuts = (np.cumsum(rng_global.dirichlet([ALPHA] * n_clients)) * len(ix)).astype(int)[:-1]
            for i, ch in enumerate(np.split(ix, cuts)): cl[i].extend(ch.tolist())
        return [np.sort(np.array(c, dtype=int)) for c in cl]
    # natural: group identifiers (service / capture file / destination host) hashed to silos
    groups = pd.Series(gtr).astype(str)
    uniq = groups.unique()
    if len(uniq) <= n_clients:                      # e.g. CIC-IDS: one silo per capture file
        return [np.where(groups.values == u)[0] for u in uniq]
    sizes = groups.value_counts()
    load = np.zeros(n_clients); assign = {}
    for g, s in sizes.items():                      # greedy size balancing of whole groups
        j = int(np.argmin(load)); assign[g] = j; load[j] += s
    a = groups.map(assign).values
    return [np.where(a == j)[0] for j in range(n_clients)]

# ----------------------------------------------------------------------------- DP binning
def dp_bin_edges(A_clients, z):
    """Federated DP quantile sketch on asinh-transformed features.
    Each client releases fine histograms (FINE bins per feature) + N(0, (z*sqrt(nf))^2) noise
    (one record changes one cell per feature -> L2 sensitivity sqrt(nf))."""
    nf = A_clients[0].shape[1]
    grid = np.linspace(-LIM, LIM, FINE + 1)
    tot = np.zeros((nf, FINE))
    for A in A_clients:
        idx = np.clip(np.searchsorted(grid, A, side="right") - 1, 0, FINE - 1)
        h = np.stack([np.bincount(idx[:, f], minlength=FINE) for f in range(nf)]).astype(float)
        if z > 0: h += rng_global.normal(0, z * math.sqrt(nf), size=h.shape)
        tot += h
    tot = np.clip(tot, 0, None)
    edges = []
    for f in range(nf):
        c = np.cumsum(tot[f]); c = c / max(c[-1], 1e-9)
        qs = np.linspace(0, 1, NBINS + 1)[1:-1]
        e = grid[1:][np.clip(np.searchsorted(c, qs), 0, FINE - 1)]
        edges.append(np.unique(e))
    return edges, nf * FINE * 4                      # bytes uploaded per client

def apply_bins(A, edges):
    return np.stack([np.searchsorted(edges[f], A[:, f], side="right") for f in range(A.shape[1])],
                    axis=1).astype(np.uint8)

# ----------------------------------------------------------------------------- robust aggregation
def aggregate(vecs, weights, mal_mask=None):
    """vecs: list of per-client *normalised* contributions (divided by client size).
    Returns weighted aggregate (normalised) and indices kept."""
    V = np.stack(vecs); w = np.asarray(weights, float); n = len(V)
    keep = np.arange(n)
    if AGG == "multikrum" and KRUM_F > 0 and n > 2 * KRUM_F + 2:
        D = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
        k = n - KRUM_F - 2
        scores = np.sort(D + np.diag(np.full(n, np.inf)), axis=1)[:, :k].sum(1)
        keep = np.sort(np.argsort(scores)[: n - KRUM_F])
        agg = (V[keep] * w[keep, None]).sum(0) / w[keep].sum()
    elif AGG in ("cclip", "bcclip", "bkrum"):
        if AGG.startswith("b"):
            # Bucketing (Karimireddy, He & Jaggi, ICLR 2022): average random pairs of clients
            # before robust aggregation to reduce honest heterogeneity.
            perm = rng_global.permutation(n); s_b = 2
            B = [perm[i:i + s_b] for i in range(0, n, s_b)]
            V = np.stack([(V[b] * w[b, None]).sum(0) / w[b].sum() for b in B])
            w = np.array([w[b].sum() for b in B]); n = len(V)
        if AGG == "bkrum" and n > 2:
            fb = max(1, min(KRUM_F, (n - 3) // 2))
            D = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
            k = max(1, n - fb - 2)
            scores = np.sort(D + np.diag(np.full(n, np.inf)), axis=1)[:, :k].sum(1)
            kk = np.sort(np.argsort(scores)[: n - fb])
            return (V[kk] * w[kk, None]).sum(0) / w[kk].sum(), keep

        # Centred clipping (Karimireddy, He & Jaggi, ICML 2021): start from the coordinate-wise
        # median and move towards the weighted mean of clipped deviations. Heterogeneous honest
        # clients are down-weighted rather than excluded.
        c = np.median(V, axis=0)
        for _ in range(3):
            dev = V - c; nrm = np.linalg.norm(dev, axis=1)
            tau = np.median(nrm) + 1e-12
            c = c + ((dev * np.minimum(1.0, tau / np.maximum(nrm, 1e-12))[:, None]) * w[:, None]).sum(0) / w.sum()
        agg = c
    elif AGG == "trimmed" and KRUM_F > 0 and n > 2 * KRUM_F:
        S = np.sort(V, axis=0); agg = S[KRUM_F: n - KRUM_F].mean(0)
    elif AGG == "median":
        agg = np.median(V, axis=0)
    else:
        agg = (V * w[:, None]).sum(0) / w.sum()
    return agg, keep

def poison_update(honest_norm, benign_norm_list):
    """Model/update-poisoning on a normalised contribution vector."""
    if ATTACK == "signflip": return -3.0 * honest_norm
    if ATTACK == "scale":    return 10.0 * honest_norm
    if ATTACK == "alie":
        B = np.stack(benign_norm_list)
        return B.mean(0) - ALIE_Z * B.std(0)
    return honest_norm

UPDATE_ATTACKS = {"signflip", "scale", "alie"}

# ----------------------------------------------------------------------------- linear FL family
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _local_sgd(Z, y, w, epochs, lr, mu=0.0, corr=None, bs=512, rng=None):
    w0 = w.copy(); w = w.copy(); steps = 0
    Zb = np.hstack([Z, np.ones((len(Z), 1))])
    for _ in range(epochs):
        perm = rng.permutation(len(Zb))
        for s in range(0, len(Zb), bs):
            b = perm[s:s + bs]
            p = sigmoid(Zb[b] @ w); g = Zb[b].T @ (p - y[b]) / len(y[b])
            if mu: g = g + mu * (w - w0)
            if corr is not None: g = g + corr
            w -= lr * g; steps += 1
    return w, steps

def linear_fl(method, Zc, yc, mal, rounds=30, lr=0.5, epochs=1, rng=None):
    d = Zc[0].shape[1] + 1; w = np.zeros(d); n_i = np.array([len(y) for y in yc], float)
    c = np.zeros(d); ci = [np.zeros(d) for _ in Zc]
    for _ in range(rounds):
        deltas, dcs = [], []
        for i, (Z, y) in enumerate(zip(Zc, yc)):
            if len(y) == 0: deltas.append(np.zeros(d)); dcs.append(np.zeros(d)); continue
            if method == "scaffold":
                wi, K = _local_sgd(Z, y, w, epochs, lr, corr=(c - ci[i]), rng=rng)
                new_ci = ci[i] - c + (w - wi) / (K * lr)
                dcs.append(new_ci - ci[i]); ci[i] = new_ci
            else:
                wi, _ = _local_sgd(Z, y, w, epochs, lr, mu=(0.01 if method == "fedprox" else 0.0), rng=rng)
            deltas.append(wi - w)
        if ATTACK in UPDATE_ATTACKS:
            benign = [deltas[i] for i in range(len(deltas)) if not mal[i]]
            deltas = [poison_update(deltas[i], benign) if mal[i] else deltas[i] for i in range(len(deltas))]
        w = w + (np.stack(deltas) * n_i[:, None]).sum(0) / n_i.sum()
        if method == "scaffold": c = c + np.stack(dcs).mean(0)
    up = rounds * d * 4 * (2 if method == "scaffold" else 1)
    return w, up

def linear_predict(w, Z): return sigmoid(np.hstack([Z, np.ones((len(Z), 1))]) @ w)

# ----------------------------------------------------------------------------- Stage 1: DP-FedSGD
def dp_fedsgd(Zc, yc, mal, z, robust, rounds, lr, secagg=False):
    """Full-batch FedSGD on logistic regression. Per-example gradients clipped to L2 norm 1,
    summed, Gaussian noise std z*1 added by each client (record-level DP, sensitivity 1)."""
    d = Zc[0].shape[1] + 1; w = np.zeros(d); n_i = np.array([len(y) for y in yc], float)
    for _ in range(rounds):
        g_norm = []
        for i, (Z, y) in enumerate(zip(Zc, yc)):
            Zb = np.hstack([Z, np.ones((len(Z), 1))])
            r = sigmoid(Zb @ w) - y
            norms = np.abs(r) * np.linalg.norm(Zb, axis=1)
            scale = np.minimum(1.0, 1.0 / np.maximum(norms, 1e-12))
            gs = Zb.T @ (r * scale)
            if z > 0: gs = gs + rng_global.normal(0, z / (math.sqrt(len(yc)) if secagg else 1.0), size=d)
            g_norm.append(gs / max(n_i[i], 1))
        if ATTACK in UPDATE_ATTACKS:
            benign = [g_norm[i] for i in range(len(g_norm)) if not mal[i]]
            g_norm = [poison_update(g_norm[i], benign) if mal[i] else g_norm[i] for i in range(len(g_norm))]
        if robust: g, _ = aggregate(g_norm, n_i)
        else:      g = (np.stack(g_norm) * n_i[:, None]).sum(0) / n_i.sum()
        w = w - lr * g
    return w, rounds * d * 4

# ----------------------------------------------------------------------------- Stage 2: federated histogram GBDT
class Tree:
    def __init__(self):
        self.feat, self.thr, self.left, self.right, self.val, self.cover = [], [], [], [], [], []
    def add(self, cover):
        for a, v in ((self.feat, -2), (self.thr, 0), (self.left, -1), (self.right, -1), (self.val, 0.0)):
            a.append(v)
        self.cover.append(max(float(cover), 1e-6)); return len(self.feat) - 1
    def predict(self, Xb):
        feat, thr = np.array(self.feat), np.array(self.thr)
        left, right, val = np.array(self.left), np.array(self.right), np.array(self.val)
        cur = np.zeros(len(Xb), dtype=int)
        for _ in range(DEPTH + 1):
            inner = left[cur] >= 0
            if not inner.any(): break
            rows = np.where(inner)[0]; nd = cur[rows]
            go_left = Xb[rows, feat[nd]] <= thr[nd]
            cur[rows] = np.where(go_left, left[nd], right[nd])
        return val[cur]
    def to_shap(self):
        n = len(self.feat)
        cov = list(self.cover)
        for nd in range(n - 1, -1, -1):                   # children always have larger ids
            if self.left[nd] >= 0: cov[nd] = cov[self.left[nd]] + cov[self.right[nd]]
        self.cover = cov
        return {"children_left": np.array(self.left), "children_right": np.array(self.right),
                "children_default": np.array(self.left), "features": np.array(self.feat),
                "thresholds": np.array(self.thr, dtype=float) + 0.5,
                "values": np.array(self.val, dtype=float).reshape(n, 1),
                "node_sample_weight": np.array(self.cover)}

def fed_hist_gbdt(Xc, yc, F0c, mal, z, robust, nb, nf, secagg=False):
    """Level-wise histogram GBDT. Per level, every client uploads G/H histograms for its
    active nodes: (K, nf, nb) x 2 floats. Sensitivity per record: |g|<=1, h<=1/4 in each of nf
    cells -> L2 = sqrt(nf * (1 + 1/16)). Noise std z*sens added *by each client*."""
    sens = math.sqrt(nf * (1 + 1 / 16)); n_i = np.array([len(y) for y in yc], float)
    nC = len(yc)
    sd_client = z * sens / (math.sqrt(nC) if secagg else 1.0)        # per-client noise std
    sd_agg = sd_client * math.sqrt(nC)                                # std of aggregated cell
    # a split side whose hessian mass is comparable to the noise of a cumulative sum over the
    # bins is not trusted; with z = 0 this reduces to MIN_HESS
    min_hess = max(MIN_HESS, sd_agg * math.sqrt(nb))
    Fc = [f.copy() for f in F0c]
    Xoff = [X.astype(np.int64) + np.arange(nf)[None, :] * nb for X in Xc]
    trees, up_bytes, rounds, kept_log = [], 0, 0, []
    for t in range(N_TREES):
        tree = Tree(); root = tree.add(1.0)
        node_c = [np.zeros(len(y), dtype=np.int64) for y in yc]       # level-local node ids
        level_nodes = [root]
        gh = []
        for i, y in enumerate(yc):
            p = sigmoid(Fc[i]); gh.append((p - y, p * (1 - p)))
        for lvl in range(DEPTH):
            K = len(level_nodes)
            if K == 0: break
            vecs = []
            for i in range(len(yc)):
                act = node_c[i] >= 0
                g, h = gh[i]
                if act.any():
                    idx = (node_c[i][act][:, None] * (nf * nb) + Xoff[i][act]).ravel()
                    G = np.bincount(idx, weights=np.repeat(g[act], nf), minlength=K * nf * nb)
                    H = np.bincount(idx, weights=np.repeat(h[act], nf), minlength=K * nf * nb)
                else:
                    G = np.zeros(K * nf * nb); H = np.zeros(K * nf * nb)
                if z > 0:
                    G = G + rng_global.normal(0, sd_client, size=G.shape)
                    H = H + rng_global.normal(0, sd_client, size=H.shape)
                vecs.append(np.concatenate([G, H]) / max(n_i[i], 1))
            if ATTACK in UPDATE_ATTACKS:
                benign = [vecs[i] for i in range(len(vecs)) if not mal[i]]
                for i in range(len(vecs)):
                    if mal[i]:
                        if ATTACK == "alie": vecs[i] = poison_update(vecs[i], benign)
                        else:  # poison gradients only; hessians stay plausible
                            v = vecs[i].copy(); half = len(v) // 2
                            v[:half] = poison_update(v[:half], [b[:half] for b in benign]); vecs[i] = v
            up_bytes += K * nf * nb * 2 * 4; rounds += 1
            if robust: agg, keep = aggregate(vecs, n_i)
            else:      agg = (np.stack(vecs) * n_i[:, None]).sum(0) / n_i.sum(); keep = np.arange(len(vecs))
            kept_log.append(len(keep))
            tot_n = n_i.sum()
            G = (agg[: K * nf * nb] * tot_n).reshape(K, nf, nb)
            H = np.clip((agg[K * nf * nb:] * tot_n).reshape(K, nf, nb), 0, None)
            GL, HL = np.cumsum(G, axis=2)[:, :, :-1], np.cumsum(H, axis=2)[:, :, :-1]
            Gt, Ht = G.sum(2, keepdims=True), H.sum(2, keepdims=True)
            GR, HR = Gt - GL, Ht - HL
            gain = GL**2 / (HL + LAMBDA) + GR**2 / (HR + LAMBDA) - Gt**2 / (Ht + LAMBDA)
            gain[(HL < min_hess) | (HR < min_hess)] = -np.inf
            next_nodes, remap = [], {}
            for k, nd in enumerate(level_nodes):
                gk = gain[k]; best = np.unravel_index(np.argmax(gk), gk.shape)
                Gk, Hk = float(Gt[k, best[0], 0]), float(Ht[k, best[0], 0])
                last = lvl == DEPTH - 1
                if not np.isfinite(gk[best]) or gk[best] <= 0:
                    tree.val[nd] = -LR * Gk / (Hk + LAMBDA); remap[k] = None; continue
                f, b = int(best[0]), int(best[1])
                tree.feat[nd], tree.thr[nd] = f, b
                l = tree.add(HL[k, f, b]); r = tree.add(HR[k, f, b])
                tree.cover[nd] = tree.cover[l] + tree.cover[r]      # consistent covers for TreeSHAP
                tree.left[nd], tree.right[nd] = l, r
                if last:
                    tree.val[l] = -LR * float(GL[k, f, b]) / (float(HL[k, f, b]) + LAMBDA)
                    tree.val[r] = -LR * float(GR[k, f, b]) / (float(HR[k, f, b]) + LAMBDA)
                    remap[k] = None
                else:
                    remap[k] = (f, b, len(next_nodes), len(next_nodes) + 1); next_nodes += [l, r]
            for i in range(len(yc)):                                   # route records
                nc = node_c[i]; new = np.full_like(nc, -1)
                for k, info in remap.items():
                    if info is None: continue
                    f, b, li, ri = info
                    m = nc == k
                    new[m] = np.where(Xc[i][m, f] <= b, li, ri)
                node_c[i] = new
            level_nodes = next_nodes
        trees.append(tree)
        for i in range(len(yc)): Fc[i] = Fc[i] + tree.predict(Xc[i])
    return trees, up_bytes, rounds, kept_log

def gbdt_predict(trees, Xb, F0):
    F = F0.copy()
    for t in trees: F = F + t.predict(Xb)
    return F

# ----------------------------------------------------------------------------- metrics
def metrics(y, score, thr=0.5):
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
           "fpr": fp / max(fp + tn, 1)}
    try: out["auc"] = roc_auc_score(y, score)
    except Exception: out["auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}

# ----------------------------------------------------------------------------- per-dataset run
def run_dataset(name):
    t0 = time.time()
    loaded = LOADERS[name](SPLIT)
    if loaded is None: log(f"[skip {name}] no data in {DATA_ROOT/name}"); return None
    tr, ytr, te, yte, gtr, split_used = loaded
    tr, ytr, te, yte, gtr, dstats = dedup(tr, ytr, te, yte, gtr)
    feat_names = list(tr.columns)
    Atr, Ate = np.arcsinh(tr.values.astype(float)), np.arcsinh(te.values.astype(float))
    nf = Atr.shape[1]
    clients = [c for c in make_clients(ytr, gtr, N_CLIENTS) if len(c) > 0]
    nC = len(clients)
    mal = np.zeros(nC, dtype=bool)
    if N_MAL > 0 and ATTACK != "none":
        mal[rng_global.choice(nC, min(N_MAL, nC), replace=False)] = True
    log(f"\n=== {name.upper()}  split={split_used}  clients={nC}  nf={nf}  "
        f"train={len(ytr)}  test={len(yte)}  attack={ATTACK} x{int(mal.sum())}")

    releases = 1 + T1 + N_TREES * DEPTH
    z = z_for_epsilon(EPS, releases, DELTA)
    eps_achieved = rdp_epsilon(z, releases, DELTA) if z > 0 else float("inf")
    log(f"  DP: target eps={EPS} delta={DELTA} releases={releases} noise multiplier z={z:.3f} achieved eps={eps_achieved:.3f}")

    # client data views
    yc = [ytr[c].copy() for c in clients]
    Ac = [Atr[c] for c in clients]
    # DP sketch (used by every federated method so all share the same binning).
    edges, sketch_bytes = dp_bin_edges(Ac, z)
    nb = max(len(e) for e in edges) + 1
    Xc = [apply_bins(A, edges) for A in Ac]
    Xte = apply_bins(Ate, edges)
    # data poisoning
    # Trigger feature is fixed from clean data in *every* run, so the clean-model trigger
    # misclassification rate (baseline ASR) is always reported alongside the attacked one.
    top = np.stack([(X == nb - 1).mean(0) for X in Xc]).mean(0)
    trig = int(np.argmin(top)); trig_bin = int(nb - 1)
    if ATTACK == "labelflip":
        for i in np.where(mal)[0]: yc[i] = 1 - yc[i]
    if ATTACK == "backdoor":
        for i in np.where(mal)[0]:
            att = np.where(yc[i] == 1)[0]
            sel = rng_global.choice(att, len(att) // 2, replace=False) if len(att) else att
            Xc[i] = Xc[i].copy(); Xc[i][sel, trig] = trig_bin; yc[i][sel] = 0
    Zc = [X.astype(float) / (nb - 1) for X in Xc]; Zte = Xte.astype(float) / (nb - 1)
    Xte_trig = None
    if trig is not None:
        Xte_trig = Xte[yte == 1].copy(); Xte_trig[:, trig] = nb - 1

    res, extra = {}, {}
    def record(mname, score, t_start, up_bytes=0, rounds=0, score_trig=None, **kw):
        m = metrics(yte, score); m.update(train_seconds=time.time() - t_start,
                                          uplink_MB_per_client=up_bytes / 1e6, rounds=rounds, **kw)
        if score_trig is not None: m["backdoor_ASR"] = float((score_trig < 0.5).mean())
        res[mname] = m
        log(f"  {mname:22s} F1={m['f1']:.4f} AUC={m['auc']:.4f} up={m['uplink_MB_per_client']:.2f}MB"
            + (f" ASR={m['backdoor_ASR']:.3f}" if 'backdoor_ASR' in m else ""))

    # ---- centralised references (pooled clean training data; upper bounds, not federated)
    sc = StandardScaler().fit(Atr); Str, Ste = sc.transform(Atr), sc.transform(Ate)
    for mname in [m for m in RUN_METHODS if m.startswith("central_")]:
        t = time.time()
        if mname == "central_logreg":
            mdl = LogisticRegression(max_iter=1000).fit(Str, ytr); s = mdl.predict_proba(Ste)[:, 1]
        elif mname == "central_rf":
            mdl = RandomForestClassifier(200, n_jobs=-1, random_state=SEED).fit(Atr, ytr); s = mdl.predict_proba(Ate)[:, 1]
        elif mname == "central_xgb":
            mdl = xgb.XGBClassifier(n_estimators=N_TREES, max_depth=DEPTH, learning_rate=LR, reg_lambda=LAMBDA,
                                    max_bin=NBINS, tree_method="hist", verbosity=0, n_jobs=-1,
                                    random_state=SEED).fit(Atr, ytr); s = mdl.predict_proba(Ate)[:, 1]
        elif mname == "central_lgbm_matched":
            mdl = lgb.LGBMClassifier(n_estimators=N_TREES, max_depth=DEPTH, num_leaves=2 ** DEPTH,
                                     learning_rate=LR, reg_lambda=LAMBDA, max_bin=NBINS, verbosity=-1,
                                     random_state=SEED, n_jobs=-1).fit(Atr, ytr); s = mdl.predict_proba(Ate)[:, 1]
        else:
            mdl = lgb.LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, verbosity=-1,
                                     random_state=SEED, n_jobs=-1).fit(Atr, ytr); s = mdl.predict_proba(Ate)[:, 1]
        record(mname, s, t)

    # ---- linear federated baselines (non-private references)
    for mname in [m for m in RUN_METHODS if m in ("fedavg", "fedprox", "scaffold")]:
        t = time.time()
        w, up = linear_fl(mname, Zc, yc, mal, rng=rng_global)
        st = None
        if Xte_trig is not None: st = linear_predict(w, Xte_trig.astype(float) / (nb - 1))
        record(mname, linear_predict(w, Zte), t, up, 30, st)

    # ---- federated XGBoost ensemble (each silo trains locally, server averages probabilities)
    if "fedxgb_ens" in RUN_METHODS:
        t = time.time(); P = np.zeros(len(yte)); PT = None if Xte_trig is None else np.zeros(len(Xte_trig))
        W, up = 0.0, 0
        for i in range(nC):
            if len(np.unique(yc[i])) < 2: continue
            mdl = xgb.XGBClassifier(n_estimators=N_TREES, max_depth=DEPTH, learning_rate=LR, tree_method="hist",
                                    max_bin=NBINS, verbosity=0, n_jobs=-1, random_state=SEED).fit(Xc[i], yc[i])
            p = mdl.predict_proba(Xte)[:, 1]; pt = None if PT is None else mdl.predict_proba(Xte_trig)[:, 1]
            if mal[i] and ATTACK in UPDATE_ATTACKS:
                p = 1 - p; pt = None if pt is None else 1 - pt
            P += len(yc[i]) * p; W += len(yc[i]); up = max(up, len(mdl.get_booster().save_raw("ubj")))
            if PT is not None: PT += len(yc[i]) * pt
        record("fedxgb_ens", P / W, t, up, 1, None if PT is None else PT / W)

    # ---- histogram-GBDT family
    def gbdt_variant(mname, warm, dp, robust, secagg=False):
        t = time.time(); zz = z if dp else 0.0; up = sketch_bytes; rounds = 1
        if warm:
            w, b1 = dp_fedsgd(Zc, yc, mal, zz, robust, T1, LR1, secagg); up += b1; rounds += T1
            F0c = [linear_predict_logit(w, Z) for Z in Zc]
            F0te = linear_predict_logit(w, Zte)
            F0tt = None if Xte_trig is None else linear_predict_logit(w, Xte_trig.astype(float) / (nb - 1))
        else:
            w = None
            F0c = [np.zeros(len(y)) for y in yc]; F0te = np.zeros(len(yte))
            F0tt = None if Xte_trig is None else np.zeros(len(Xte_trig))
        trees, b2, r2, kept = fed_hist_gbdt(Xc, yc, F0c, mal, zz, robust, nb, nf, secagg)
        up += b2; rounds += r2
        Fte = gbdt_predict(trees, Xte, F0te)
        st = None if F0tt is None else sigmoid(gbdt_predict(trees, Xte_trig, F0tt))
        record(mname, sigmoid(Fte), t, up, rounds, st,
               epsilon=(eps_achieved if dp else float("inf")), mean_clients_kept=float(np.mean(kept)))
        return trees, w, F0te

    variants = {"fedhist": (False, False, False, False), "ours": (True, True, True, False),
                "ours_nowarm": (False, True, True, False), "ours_nodp": (True, False, True, False),
                "ours_norobust": (True, True, False, False), "ours_secagg": (True, True, False, True)}
    ours_model = None
    for mname, (warm, dp, robust, sa) in variants.items():
        if mname in RUN_METHODS:
            try:
                out = gbdt_variant(mname, warm, dp, robust, sa)
                if mname == "ours": ours_model = out
            except Exception as e:
                log(f"  [error {mname}] {e}"); traceback.print_exc()

    # ---- SHAP for the federated model
    if DO_SHAP and ours_model is not None:
        try:
            import shap
            trees, w, _ = ours_model
            idx = rng_global.choice(len(Xte), min(2000, len(Xte)), replace=False)
            ex = shap.TreeExplainer({"trees": [t.to_shap() for t in trees], "base_offset": 0.0})
            sv = np.asarray(ex.shap_values(Xte[idx].astype(float)))
            if w is not None:
                Zs = Zte[idx]; sv = sv + w[:-1][None, :] * (Zs - Zs.mean(0))[:, :]
            imp = np.abs(sv).mean(0); order = np.argsort(-imp)[:15]
            extra["shap_top15"] = [{"feature": feat_names[j], "mean_abs_shap": float(imp[j])} for j in order]
        except Exception as e:
            log(f"  [shap error] {e}")

    extra.update(split=split_used, data=dstats, n_clients=nC, n_features=nf, n_bins=int(nb),
                 client_sizes=[int(len(c)) for c in clients],
                 client_pos_rate=[float(ytr[c].mean()) for c in clients],
                 malicious=[int(i) for i in np.where(mal)[0]], backdoor_trigger_feature=(None if trig is None else feat_names[trig]),
                 dp={"target_eps": EPS, "achieved_eps": eps_achieved, "delta": DELTA, "noise_multiplier": z,
                     "releases": releases, "accountant": "RDP Gaussian, Balle et al. 2020 conversion"},
                 wall_seconds=time.time() - t0)
    return {"results": res, "info": extra}

def linear_predict_logit(w, Z): return np.hstack([Z, np.ones((len(Z), 1))]) @ w

def main():
    cfg = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list))}
    cfg["DATA_ROOT"] = str(DATA_ROOT)
    log(f"[config] {json.dumps({k: cfg[k] for k in ['SEED','DATASETS','SPLIT','PARTITION','ATTACK','N_MAL','AGG','EPS_S','N_TREES','DEPTH','TAG']})}")
    out = {"config": cfg, "datasets": {}}
    t0 = time.time()
    for ds in DATASETS:
        if ds not in LOADERS: continue
        try:
            r = run_dataset(ds)
            if r is not None: out["datasets"][ds] = r
        except Exception as e:
            log(f"[error {ds}] {e}"); traceback.print_exc()
    out["wall_seconds"] = time.time() - t0
    fp = RUNS / f"{TAG}.json"
    tmp = fp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=float); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, fp)
    log(f"\nDONE in {out['wall_seconds']:.1f}s -> {fp}")

if __name__ == "__main__":
    main()
