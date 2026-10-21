#!/usr/bin/env bash
# Generates batches/*.sh for the revision sweep. Each batch is one run of fedxai_v5.py.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f batches/*.sh
mk() { # mk TAG "ENV=VAL ..."
  printf '#!/usr/bin/env bash\ncd "$(dirname "$0")/.."\n%s TAG=%s python3 fedxai_v5.py\n' "$2" "$1" > "batches/$1.sh"
}
SEEDS10="42 7 2026 11 23 31 47 53 61 73"; SEEDS3="42 7 2026"
# A. main comparison + multi-seed ablations (R2.5, R2.6, R2.8, R4.iv)
for s in $SEEDS10; do mk "A_main_s$s" "SEED=$s"; done
# B. privacy-utility curve (R2.3, R4.iii)
for e in 1 2 4 16 inf; do for s in $SEEDS3; do
  mk "B_eps${e}_s$s" "SEED=$s EPS=$e METHODS=fedhist,ours,ours_nowarm SHAP=0"; done; done
# C. attacks incl. statistics-preserving ALIE and backdoor (R2.4, R4.i, R4.ii)
for a in labelflip signflip scale alie backdoor; do for m in 1 2; do for s in $SEEDS3; do
  mk "C_${a}_m${m}_s$s" "SEED=$s ATTACK=$a N_MAL=$m METHODS=fedavg,fedxgb_ens,fedhist,ours,ours_norobust SHAP=0"; done; done; done
# C2. alternative robust aggregators
for g in multikrum trimmed median; do for a in none signflip alie backdoor; do for s in $SEEDS3; do
  nm=2; [ "$a" = none ] && nm=0
  mk "C2_${g}_${a}_s$s" "SEED=$s AGG=$g ATTACK=$a N_MAL=$nm METHODS=ours SHAP=0"; done; done; done
# D. client heterogeneity: IID and Dirichlet vs natural partition (R4.vi)
for s in $SEEDS3; do
  mk "D_iid_s$s"   "SEED=$s PARTITION=iid METHODS=central_lgbm_matched,fedavg,scaffold,fedxgb_ens,fedhist,ours SHAP=0"
  mk "D_dir01_s$s" "SEED=$s PARTITION=dirichlet ALPHA=0.1 METHODS=central_lgbm_matched,fedavg,scaffold,fedxgb_ens,fedhist,ours SHAP=0"
  mk "D_dir05_s$s" "SEED=$s PARTITION=dirichlet ALPHA=0.5 METHODS=central_lgbm_matched,fedavg,scaffold,fedxgb_ens,fedhist,ours SHAP=0"
done
# E. random split (to quantify optimism of random splits, R2.7 / R4.v)
for s in $SEEDS3; do mk "E_random_s$s" "SEED=$s SPLIT=random METHODS=central_lgbm_matched,fedavg,fedxgb_ens,fedhist,ours SHAP=0"; done
# F. hyper-parameter sensitivity (R2.6)
mk F_depth3  "SEED=42 DEPTH=3 METHODS=central_lgbm_matched,fedhist,ours SHAP=0"
mk F_depth7  "SEED=42 DEPTH=7 METHODS=central_lgbm_matched,fedhist,ours SHAP=0"
mk F_trees25 "SEED=42 N_TREES=25 METHODS=central_lgbm_matched,fedhist,ours SHAP=0"
mk F_trees100 "SEED=42 N_TREES=100 METHODS=central_lgbm_matched,fedhist,ours SHAP=0"
# G. CIC-IDS scenario split: train Mon-Thu, test Friday captures (unseen attack families)
for s in $SEEDS3; do mk "G_scenario_s$s" "SEED=$s DATASETS=cicids SPLIT=scenario METHODS=central_lgbm_matched,fedavg,fedxgb_ens,fedhist,ours SHAP=0"; done
chmod +x batches/*.sh
echo "Generated $(ls batches/*.sh | wc -l | tr -d ' ') batches"
