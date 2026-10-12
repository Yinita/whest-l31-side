#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_114_r02_antithetic_pilot.json; do sleep 5; done
uv run --frozen python experiments/phase2/full_pool_shape_resource_audit.py
