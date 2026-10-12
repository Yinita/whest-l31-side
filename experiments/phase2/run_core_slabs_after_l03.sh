#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_098_r03_fold0_mini1_local.json; do sleep 5; done
uv run --frozen python benchmark/phase2.py run --estimator estimators/ablation_v30_017_r02/estimator.py --dataset benchmark/data/mini_first8 --n-mlps 8 --runner local --output results/phase2/ablation_v30_017_r02_mini8_local.json
uv run --frozen python benchmark/phase2.py run --estimator estimators/ablation_v30_018_r02/estimator.py --dataset benchmark/data/mini_first8 --n-mlps 8 --runner local --output results/phase2/ablation_v30_018_r02_mini8_local.json
