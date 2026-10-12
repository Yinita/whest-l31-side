#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_082_r06_paired_v29_mini8.json; do sleep 5; done
uv run --frozen python benchmark/run_rpc.py --estimator estimators/ablation_v30_082_gaussian_subset_r02/estimator.py --n-mlps 8 --output results/phase2/ablation_v30_082_r05_rpc_predictions.json
uv run --frozen python benchmark/rpc_score.py --predictions results/phase2/ablation_v30_082_r05_rpc_predictions.json --dataset benchmark/data/mini_first8 --local results/phase2/ablation_v30_082_r03_gaussian_subset_mini8.json --output results/phase2/ablation_v30_082_r05_rpc_scored.json
