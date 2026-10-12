#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_098_r04_full_charged_cv.json; do sleep 5; done
uv run --frozen python benchmark/phase2.py run --estimator estimators/k3_v29/estimator.py --dataset benchmark/data/mini_shard01 --n-mlps 8 --runner local --output results/phase2/ablation_v30_116_r01_paired_v29_public.json
uv run --frozen python benchmark/phase2.py run --estimator estimators/ablation_v30_048/estimator.py --dataset benchmark/data/mini_shard01 --n-mlps 8 --runner local --output results/phase2/ablation_v30_116_r01_h18_public.json
uv run --frozen python benchmark/phase2.py compare results/phase2/ablation_v30_116_r01_paired_v29_public.json results/phase2/ablation_v30_116_r01_h18_public.json --output results/phase2/ablation_v30_116_r01_public_comparison.json
