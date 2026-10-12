#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_096_r03_samecycle_features_manifest.json; do sleep 5; done
uv run --frozen python benchmark/phase2.py run --estimator estimators/ablation_v30_035_zeroalpha_subset_r02/estimator.py --dataset benchmark/data/mini_first8 --n-mlps 1 --runner local --output results/phase2/ablation_v30_035_r02_zeroalpha_subset_mini1.json
