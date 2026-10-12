#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while kill -0 192551 2>/dev/null; do sleep 5; done
python3 - <<'PY'
import json
r=json.load(open('results/phase2/discussion_leaf8_r02_mini8.json'))['results']
assert r['n_failed_mlps']==0
assert r['adjusted_final_layer_score']<5.778979755766664e-9
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python benchmark/phase2.py run --estimator estimators/discussion_combined_r01/estimator.py --dataset benchmark/data/mini_first8 --n-mlps 1 --runner local --output results/phase2/discussion_combined_r01_mini1.json
