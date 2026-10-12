#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/l03_fold7_charged_candidate.json; do sleep 5; done
uv run --frozen python experiments/phase2/collect_l03_charged_cv.py
uv run --frozen python experiments/phase2/update_ablation_results.py
