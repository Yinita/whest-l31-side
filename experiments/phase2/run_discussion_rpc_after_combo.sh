#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while pgrep -f '^/home/sti/Desktop/projects/whest-l31-side/.venv/bin/python3 /home/sti/Desktop/projects/whest-l31-side/.venv/bin/whest run --estimator /home/sti/Desktop/projects/whest-l31-side/estimators/discussion_combined_r01/estimator.py' >/dev/null; do sleep 5; done
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python benchmark/run_rpc.py --estimator estimators/discussion_leaf8_r01/estimator.py --n-mlps 8 --output results/phase2/discussion_leaf8_r03_rpc_predictions.json
uv run --frozen python benchmark/rpc_score.py --predictions results/phase2/discussion_leaf8_r03_rpc_predictions.json --local results/phase2/discussion_leaf8_r02_mini8.json --output results/phase2/discussion_leaf8_r03_rpc_scored.json
