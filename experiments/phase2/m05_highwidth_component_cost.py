"""Measure the frozen, full M05 float64 component at increasing widths.

This is a billed math/local cost diagnostic, not an official resource pass or a
float32 competition estimator. Rejected ancestors also cost a full component.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import resource
import time
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
    report = {
        "version": "paper.M05.r07.cost", "complete": False, "rows": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method_sha256": hashlib.sha256(Path("experiments/phase2/m05_small_split_ut.py").read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ("flopscope", "whestbench", "numpy")},
        "limits": {"flop_budget": 2**41, "component_wall_limit_s": 120, "threads": 1},
        "scope": "One full float64 component; local math cost diagnosis, not RPC/official sandbox validation",
        "limitations": ["No float32 cost equivalence", "No target or benchmark score",
                        "Full adaptive tree also pays for rejected ancestors and split/final-mixture operations",
                        "Local process memory is not the official client/server memory allocation"],
    }

    def save():
        temporary = output.with_suffix(".progress.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    for n in (128, 256, 512, 1024):
        seed = 61013200 + n
        rng = np.random.default_rng(seed)
        weights = rng.normal(size=(16, n, n)) * np.sqrt(2 / n)
        mean = rng.normal(size=n) * .1
        row = {"width": n, "depth": 16, "seed": seed, "dtype": "float64", "UT_points": 2*n+1,
               "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(), "failure": None}
        started = time.monotonic()
        ctx = None
        try:
            with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=120, quiet=True) as ctx:
                prediction, covariance, criterion, _, _ = component(
                    fnp.asarray(mean), fnp.eye(n, dtype=fnp.float64),
                    [fnp.asarray(w) for w in weights], "paper_sum")
            row["finite"] = bool(np.isfinite(np.asarray(prediction)).all() and np.isfinite(np.asarray(covariance)).all())
            row["criterion"] = criterion
            if not row["finite"]:
                row["failure"] = "nonfinite"
        except Exception as exc:
            row["failure"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_s"] = time.monotonic() - started
        row["process_peak_rss_kib_cumulative"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if ctx is not None:
            row["legal_cost"] = ctx.summary_dict(by_namespace=True)
            cost = row["legal_cost"]["flops_used"]
            row["budget_fraction"] = cost / 2**41
            if not row["failure"]:
                row["same_cost_component_count_before_split_overhead"] = int(2**41 // cost)
        report["rows"].append(row)
        save()
        print(n, row.get("legal_cost", {}).get("flops_used"), row["elapsed_s"], row["failure"], flush=True)
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
