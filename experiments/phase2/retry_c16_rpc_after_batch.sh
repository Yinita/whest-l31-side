#!/usr/bin/env bash
set -eu
cd /home/sti/Desktop/projects/whest-l31-side
uv run --frozen python benchmark/run_rpc.py --estimator estimators/ablation_v30_016_r03/estimator.py --n-mlps 8 --output results/phase2/ablation_v30_016_r05_mini8_rpc.json
uv run --frozen python benchmark/rpc_score.py --predictions results/phase2/ablation_v30_016_r05_mini8_rpc.json --local results/phase2/ablation_v30_016_r03_mini8_local.json --output results/phase2/ablation_v30_016_r05_mini8_rpc_scored.json
