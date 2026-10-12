#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_030_r02_c01_mini8_subprocess.json; do sleep 5; done
while test ! -s results/phase2/ablation_v30_089_distribution_shift.json; do sleep 5; done
uv run --frozen python experiments/phase2/collect_serial_followups.py
uv run --frozen python experiments/phase2/update_ablation_results.py
