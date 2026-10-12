#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while kill -0 201359 2>/dev/null; do sleep 5; done
python3 - <<'PY'
import json
assert json.load(open('results/phase2/discussion_combined_r02_mini8.json'))['results']
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python benchmark/phase2.py run --estimator estimators/discussion_pair12_r01/estimator.py --dataset benchmark/data/mini_first8 --n-mlps 1 --runner local --output results/phase2/discussion_pair12_r01_mini1.json
