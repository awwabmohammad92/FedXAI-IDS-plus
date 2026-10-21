#!/usr/bin/env bash
cd "$(dirname "$0")/.."
SEED=42 DATASETS=nslkdd N_TREES=10 TAG=sanity python3 fedxai_v5.py && echo "SANITY OK"
