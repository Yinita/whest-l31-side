#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_116_r01_public_comparison.json; do sleep 5; done
uv run --frozen python experiments/phase2/collect_h18_public_confirmation.py
uv run --frozen python experiments/phase2/update_ablation_results.py
