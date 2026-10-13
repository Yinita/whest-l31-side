"""Prespecified strided source-column quadrature, no target or score access."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from real_source_full_k211 import KEYS
from source_cap_structured import capped_structured_source


def select_columns(values, step):
    """Same column indices for all dependent legs; no approximation of rows."""
    result = []
    for key, value in zip(KEYS, values):
        if key in ("a", "p"):
            value = value[:, ::step].copy()
        elif key in ("ell", "r1", "r2", "w2", "s", "e", "c1", "c2"):
            value = value[::step].copy()
        result.append(value)
    return result


def main():
    output = Path("results/phase2/ablation_v30_067_r23_source_columns.json")
    assert not output.exists()
    mp = Path("results/phase2/ablation_v30_067_r18_all_source_capture.json")
    meta = json.loads(mp.read_text())
    sp = Path(meta["capture"])
    assert hashlib.sha256(sp.read_bytes()).hexdigest() == meta["capture_sha256"]
    with np.load(sp) as state:
        mu, sigma, alpha = [state[k].copy() for k in ("pk1v", "sigma", "alpha")]
    sources = []
    for path, sha in meta["source_files_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
        with np.load(path) as src:
            sources.append([src[k].copy() for k in KEYS])
    rows, baseline = [], None
    for step in (1, 2, 4, 8):
        # Offline cache selection is timed separately from charged evaluation.
        # Identical simple slices can be performed on legal production operands;
        # mapping to a production estimator is still unverified.
        selected = [select_columns(vals, step) for vals in sources]
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            fm, sd, al = map(fnp.asarray, (mu, sigma, alpha))
            gate = flops.stats.norm.cdf(al).astype(fnp.float32)
            density = flops.stats.norm.pdf(al).astype(fnp.float32)
            u = 2*sd*(density+al*gate)-2*fm*gate
            total = fnp.zeros((1024, 1024), dtype=fnp.float32)
            for vals in selected:
                cap = capped_structured_source(*map(fnp.asarray, vals), u, gate)
                total = total+cap if step == 1 else total+step*cap
        out = np.asarray(total).astype(np.float64)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out-out.T)) == 0 and np.max(np.abs(np.diag(out))) == 0
        if baseline is None:
            baseline = out
        rows.append({"source_columns_each": 1024//step, "source_slots": 14,
                     "deterministic_column_stride": step, "column_offset": 0, "weight": step,
                     "relative_error_vs_full_orbit": float(np.linalg.norm(out-baseline)/np.linalg.norm(baseline)),
                     "max_abs_delta": float(np.max(np.abs(out-baseline))),
                     "orbit_norm": float(np.linalg.norm(out)),
                     "projected_core_delta_norm": float(np.linalg.norm(out-baseline)*6/1028),
                     "legal_cost": ctx.summary_dict(),
                     "budget_fraction": ctx.summary_dict()["flops_used"]/2**41})
        print({k:v for k,v in rows[-1].items() if k!='legal_cost'}, flush=True)
    ref = json.loads(Path("results/phase2/ablation_v30_067_r21_structured_source.json").read_text())
    assert rows[0]["legal_cost"]["flops_used"] == ref["before_after_costs"][1]["legal_cost"]["flops_used"]
    output.write_text(json.dumps({"version": "v30.067.r23", "complete": True, "rows": rows,
                                 "before": "All14sources/all1024columns/exactstructured orbit",
                                 "after": "Fixed strided column quadrature512/256/128, same allsource scope",
                                 "single_change": "source-column quadrature count only; scale by stride",
                                 "dataset_commit": meta["dataset_commit"], "network_id": meta["network_id"], "layer": 14,
                                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "dependency_sha256": hashlib.sha256(Path("experiments/phase2/source_cap_structured.py").read_bytes()).hexdigest(),
                                 "capture_metadata_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
                                 "no_adjusted_score": True, "not_resource_acceptance": True,
                                 "limitations": ["Prespecified offset only, no unbiasedness or independent-network claim",
                                                 "Component bias not final MSE; no submitted source selection",
                                                 "Complete diagrams/proxyremoval/production slicing still unverified"]},
                                indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
