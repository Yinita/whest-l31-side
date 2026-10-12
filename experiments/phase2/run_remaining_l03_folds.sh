#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_018_r02_mini8_local.json; do sleep 5; done
for held in 1 2 3 4 5 6 7; do
 uv run --frozen python benchmark/phase2.py run --estimator estimators/k3_v29/estimator.py --dataset "benchmark/data/l03_singletons/network_${held}" --n-mlps 1 --runner local --output "results/phase2/l03_fold${held}_paired_v29_control.json"
 uv run --frozen python benchmark/phase2.py run --estimator "estimators/ablation_v30_098_fold${held}/estimator.py" --dataset "benchmark/data/l03_singletons/network_${held}" --n-mlps 1 --runner local --output "results/phase2/l03_fold${held}_charged_candidate.json"
done
