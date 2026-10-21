#!/usr/bin/env bash
# Resume-safe: skips any batch whose runs/<TAG>.json already exists.
cd "$(dirname "$0")/.."
for b in batches/*.sh; do
  tag=$(basename "$b" .sh)
  if [ -f "runs_r1/$tag.json" ]; then echo "skip $tag"; continue; fi
  echo "=== $tag ($(date '+%H:%M')) ==="
  bash "$b" || echo "!! $tag failed"
done
echo "=== ALL DONE ==="; ls runs_r1/*.json | wc -l
