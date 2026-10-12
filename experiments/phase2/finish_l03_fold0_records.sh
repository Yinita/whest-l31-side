#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_098_r03_fold0_mini1_local.json; do sleep 5; done
uv run --frozen python experiments/phase2/collect_l03_fold0.py
uv run --frozen python experiments/phase2/update_ablation_results.py
