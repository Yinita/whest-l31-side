#!/usr/bin/env bash
# Mini sweep: pure RQMC at C/B ∈ {0.10, 0.20, 0.40, 0.70}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT_DIR="${ROOT}/experiments/results/pure010s_sweep"
mkdir -p "$OUT_DIR"

DATASET="hf://aicrowd/arc-whestbench-public-2026@v1-phase1"

run_one() {
  local tag="$1"
  local est="$2"
  local log="${OUT_DIR}/${tag}.log"
  echo "=== ${tag} :: ${est} ==="
  uv run whest run --estimator "${est}" \
    --dataset "${DATASET}" \
    --split mini --runner local \
    2>&1 | tee "${log}"
  echo "--- done ${tag} ---"
}

run_one u10 experiments/pure010s_u10.py
run_one u20 experiments/pure010s_u20.py
run_one u40 experiments/pure010s_u40.py
run_one u70 experiments/pure010s_u70.py

echo "Sweep complete. Logs in ${OUT_DIR}"
