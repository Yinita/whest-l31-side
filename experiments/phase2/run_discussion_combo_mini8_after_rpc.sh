#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while pgrep -f '^/home/sti/Desktop/projects/whest-l31-side/.venv/bin/python3 benchmark/run_rpc.py --estimator estimators/discussion_leaf8_r01/estimator.py' >/dev/null; do sleep 5; done
python3 - <<'PY'
import json,time
from pathlib import Path
p=Path('results/phase2/discussion_leaf8_r03_rpc_scored.json')
for attempt in range(120):
    try:
        assert json.loads(p.read_text())
        break
    except (FileNotFoundError,json.JSONDecodeError):
        time.sleep(1)
else:
    raise RuntimeError('RPC scoring report absent; do not start next experiment')
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python benchmark/phase2.py run --estimator estimators/discussion_combined_r01/estimator.py --dataset benchmark/data/mini_first8 --n-mlps 8 --runner local --output results/phase2/discussion_combined_r02_mini8.json
