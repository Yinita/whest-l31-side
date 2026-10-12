#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while pgrep -f '^/home/sti/Desktop/projects/whest-l31-side/.venv/bin/python3 (experiments/phase2/execute_ready_batch.py|benchmark/run_rpc.py)' >/dev/null; do sleep 5; done
uv run --frozen python experiments/phase2/reference_cpu_pilot.py
