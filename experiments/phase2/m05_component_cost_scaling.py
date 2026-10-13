"""Freeze width scaling of ONE complete M05 component, with real legal meters.

Component counts include rejected ancestors in an adaptive tree. The covariance
and PSD-root costs are part of this gate; no scalar-forward-only cost claim.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from m05_small_split_ut import component


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    rows = []
    for n in (8, 16, 32, 64):
        seed = 61013100 + n
        rng = np.random.default_rng(seed)
        weights = rng.normal(size=(16, n, n)) * np.sqrt(2 / n)
        mean = rng.normal(size=n) * .1
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            prediction, covariance, criterion, _, _ = component(
                fnp.asarray(mean), fnp.eye(n, dtype=fnp.float64),
                [fnp.asarray(w) for w in weights], "paper_sum")
        summary = ctx.summary_dict(by_namespace=True)
        assert np.isfinite(np.asarray(prediction)).all()
        assert np.isfinite(np.asarray(covariance)).all()
        rows.append({"width": n, "depth": 16, "seed": seed, "dtype": "float64",
                     "UT_points": 2 * n + 1, "criterion": criterion,
                     "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
                     "legal_cost": summary,
                     "full_budget_max_component_evaluations_at_measured_small_width": int((2**41) // summary["flops_used"]),
                     "analytic_forward_only_flops": 2 * (2 * n + 1) * 16 * n**2,
                     "notes": "one component, no recursive split; forward, Taylor, covariances, W2 and roots all billed"})
        print(n, summary["flops_used"], summary["wall_time_s"], flush=True)
    result = {"version": "paper.M05.r06.cost", "complete": True, "rows": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "method_sha256": hashlib.sha256(Path("experiments/phase2/m05_small_split_ut.py").read_bytes()).hexdigest(),
              "versions": {p: importlib.metadata.version(p) for p in ("flopscope", "whestbench", "numpy")},
              "scope": "Smallwidth fullcomponent meter scaling, no score/reference; no1024wide resource or flops extrapolation acceptance",
              "limitations": ["float64 rates differ from float32 competitioncandidate",
                              "adaptive tree needs ancestor evaluations in addition to terminal components",
                              "no silent covariance/subspace/rank drop",
                              "measured smallwidth budget fits do not prove highwidth fit"]}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
