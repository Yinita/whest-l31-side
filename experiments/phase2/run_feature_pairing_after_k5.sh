#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_082_r05_rpc_scored.json; do sleep 5; done
uv run --frozen python -m experiments.phase2.cache_v29_features_samecycle
