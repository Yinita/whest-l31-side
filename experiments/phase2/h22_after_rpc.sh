#!/usr/bin/env bash
set -eu
cd /home/sti/Desktop/projects/whest-l31-side
while pgrep -f '^/home/sti/Desktop/projects/whest-l31-side/.venv/bin/python3 benchmark/run_rpc.py.*ablation_v30_016_r05' >/dev/null; do sleep 5; done
uv run --frozen python experiments/phase2/execute_ready_batch.py --keys H22 --n-mlps 8 --revision r02
uv run --frozen python experiments/phase2/update_ablation_results.py
