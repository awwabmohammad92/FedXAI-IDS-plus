# FedXAI-IDS+

Code for *FedXAI-IDS+: Differentially Private and Byzantine-Robust Federated Gradient Boosting for Cross-Silo Intrusion Detection*.

The repository reproduces every experiment in the paper: a two-stage federated pipeline (DP federated warm start + federated histogram gradient boosting), record-level (ε, δ)-DP accounting for every upload, robust aggregation (centred clipping, Multi-Krum, trimmed mean, median), five attacks (label flip, sign flip, scaling, ALIE, backdoor), and centralised and federated baselines.

## Data
Datasets are not redistributed. Download them from their providers and place them as follows:

| Dataset | Files | Folder |
|---|---|---|
| NSL-KDD | `KDDTrain+.txt`, `KDDTest+.txt` (rename to `KDDTrain.txt`, `KDDTest.txt`) | `data/nslkdd/` |
| CIC-IDS-2017 | the eight `MachineLearningCVE` CSV files | `data/cicids/` |
| TON_IoT | `train_test_network.csv` | `data/toniot/` |

## Run
```bash
pip install -r requirements.txt
bash scripts/sanity.sh            # quick NSL-KDD check
bash scripts/make_batches.sh      # writes the experiment grid to batches/
bash scripts/run_all.sh           # runs every batch; resumes if interrupted
python3 scripts/analyze_v5.py     # tables/summary.csv and tables/tests.csv
```
Each run writes `runs_r1/<TAG>.json` with the configuration, data statistics (duplicates removed, split sizes, per-client sizes and attack rates), the privacy parameters (noise multiplier, achieved ε), per-method metrics, uplink bytes and SHAP attributions.

Key options (environment variables): `DATASETS`, `SPLIT` (default / random / scenario), `PARTITION` (natural / iid / dirichlet), `EPS`, `DP_MODE` (local / secagg), `AGG` (cclip / multikrum / trimmed / median / mean), `ATTACK`, `N_MAL`, `N_TREES`, `DEPTH`, `SEED`. See the header of `fedxai_v5.py`.

Hardware used in the paper: Apple M4, 16 GB RAM; the full grid takes several hours.

## Results in this repository
- `results/runs_r1/` — one JSON per run (118 runs: 10-seed main comparison, privacy sweep, attacks, aggregators, partitions, splits, sensitivity and multi-class). Each file contains the configuration, data statistics, privacy parameters, per-method metrics, uplink and SHAP attributions.
- `results/tables/summary.csv`, `results/tables/tests.csv` — means, 95% CIs, Holm-corrected Wilcoxon tests and TOST equivalence tests (produced by `scripts/analyze_v5.py`).
- `analysis/` — scripts that produce the figures in `figures/` (run from `results/`: `python3 ../analysis/make_figs.py`).

## License
MIT
