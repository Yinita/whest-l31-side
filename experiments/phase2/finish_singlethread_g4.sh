#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_102_r03_singlethread_g4_features.json; do sleep 5; done
uv run --frozen python experiments/phase2/g4_feature_screen_singlethread.py
