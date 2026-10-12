#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_030_r02_c01_mini8_subprocess.json; do sleep 5; done
uv run --frozen python benchmark/phase2.py run --estimator estimators/ablation_v30_098_fold0/estimator.py --dataset benchmark/data/mini_first8 --n-mlps 1 --runner local --output results/phase2/ablation_v30_098_r03_fold0_mini1_local.json
