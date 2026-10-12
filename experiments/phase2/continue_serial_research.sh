#!/usr/bin/env bash
set -eu
cd /home/sti/Desktop/projects/whest-l31-side
# Resume only once the identified offline trace completes, never duplicate it.
while test -e /proc/4076743/stat; do
  tick=$(awk '{print $22}' /proc/4076743/stat)
  expected=$(python3 -c 'import json;print(json.load(open("experiments/phase2/research_state.json"))["active_offline_diagnostic"]["pid_start_ticks"])')
  if test "$tick" != "$expected"; then break; fi
  sleep 5
done
uv run --frozen python benchmark/run_rpc.py --estimator estimators/ablation_v30_016_r03/estimator.py --n-mlps 8 --output results/phase2/ablation_v30_016_r04_mini8_rpc.json
uv run --frozen python benchmark/rpc_score.py --predictions results/phase2/ablation_v30_016_r04_mini8_rpc.json --local results/phase2/ablation_v30_016_r03_mini8_local.json --output results/phase2/ablation_v30_016_r04_mini8_rpc_scored.json
uv run --frozen python experiments/phase2/execute_ready_batch.py --keys H19 H20 H22 --n-mlps 1 --revision r01
uv run --frozen python experiments/phase2/update_ablation_results.py
