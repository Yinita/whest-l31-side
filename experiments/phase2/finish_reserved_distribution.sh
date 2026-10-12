#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -f results/phase2/ablation_v30_089_reserved_state_only.json; do sleep 5; done
uv run --frozen python experiments/phase2/compare_reserved_state_quantiles.py
