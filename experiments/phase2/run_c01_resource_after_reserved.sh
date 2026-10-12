#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -f results/phase2/ablation_v30_089_reserved_state_only.json; do sleep 5; done
uv run --frozen python benchmark/phase2.py run --estimator estimators/k3_v29_symrecursive/estimator.py --dataset benchmark/data/mini_first8 --n-mlps 8 --runner subprocess --output results/phase2/ablation_v30_030_r02_c01_mini8_subprocess.json
