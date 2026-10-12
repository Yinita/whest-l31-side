#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while kill -0 4110082 2>/dev/null; do sleep 5; done
uv run --frozen python benchmark/rpc_score.py --predictions results/phase2/ablation_v30_052_r04_mini8_rpc.json --local results/phase2/ablation_v30_052_r03_mini8_local.json --output results/phase2/ablation_v30_052_r04_mini8_rpc_scored.json
uv run --frozen python experiments/phase2/execute_ready_batch.py --keys C01 --n-mlps 8 --revision r02
uv run --frozen python benchmark/run_rpc.py --estimator estimators/k3_v29_symrecursive/estimator.py --n-mlps 8 --output results/phase2/ablation_v30_030_r01_c01_mini8_rpc.json
uv run --frozen python benchmark/rpc_score.py --predictions results/phase2/ablation_v30_030_r01_c01_mini8_rpc.json --local results/phase2/ablation_v30_001_r02_mini8_local.json --output results/phase2/ablation_v30_030_r01_c01_mini8_rpc_scored.json
uv run --frozen python experiments/phase2/update_ablation_results.py
