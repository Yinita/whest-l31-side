#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while pgrep -f '^/home/sti/Desktop/projects/whest-l31-side/.venv/bin/python3 (benchmark/run_rpc.py|experiments/phase2/reference_cpu_pilot.py)' >/dev/null; do sleep 5; done
while test ! -f results/phase2/ablation_v30_114_reference_cpu_pilot.json; do sleep 5; done
uv run --frozen python experiments/phase2/capture_reserved_state_only.py
