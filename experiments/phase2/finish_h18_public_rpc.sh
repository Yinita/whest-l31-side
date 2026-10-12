#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while ! uv run --frozen python - <<'PY'
import json
from pathlib import Path
p=Path('results/phase2/ablation_v30_116_r02_h18_public_rpc.json')
try:r=json.loads(p.read_text())
except (ValueError,FileNotFoundError):raise SystemExit(1)
assert len(r.get('per_mlp',[])) == 8
PY
do sleep 10; done
uv run --frozen python benchmark/rpc_score.py --dataset benchmark/data/mini_shard01 --predictions results/phase2/ablation_v30_116_r02_h18_public_rpc.json --local results/phase2/ablation_v30_116_r01_h18_public.json --output results/phase2/ablation_v30_116_r02_h18_public_rpc_scored.json
