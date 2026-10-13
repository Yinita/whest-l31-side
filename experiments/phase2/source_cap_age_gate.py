"""Target-free single-factor source-age truncation against full represented orbit."""

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


def main():
    output = Path("results/phase2/ablation_v30_067_r22_source_age.json")
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
    rows, outputs = [], []
    for keep in (14, 8, 4, 1):
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            fm, sd, al = map(fnp.asarray, (mu, sigma, alpha))
            gate = flops.stats.norm.cdf(al).astype(fnp.float32)
            density = flops.stats.norm.pdf(al).astype(fnp.float32)
            u = 2*sd*(density+al*gate)-2*fm*gate
            total = fnp.zeros((1024, 1024), dtype=fnp.float32)
            for vals in sources[-keep:]:
                total = total+capped_structured_source(*map(fnp.asarray, vals), u, gate)
        out = np.asarray(total).astype(np.float64)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out-out.T)) == 0 and np.max(np.abs(np.diag(out))) == 0
        outputs.append(out)
        baseline = outputs[0]
        rows.append({"source_slots_kept": keep, "source_indices": list(range(14-keep, 14)),
                     "relative_error_vs_full_orbit": float(np.linalg.norm(out-baseline)/np.linalg.norm(baseline)),
                     "max_abs_delta": float(np.max(np.abs(out-baseline))),
                     "orbit_norm": float(np.linalg.norm(out)),
                     "projected_core_delta_norm": float(np.linalg.norm(out-baseline)*6/1028),
                     "legal_cost": ctx.summary_dict(),
                     "budget_fraction": ctx.summary_dict()["flops_used"]/2**41})
        print({k:v for k,v in rows[-1].items() if k!='legal_cost'}, flush=True)
    reference = json.loads(Path("results/phase2/ablation_v30_067_r21_structured_source.json").read_text())
    assert rows[0]["legal_cost"]["flops_used"] == reference["before_after_costs"][1]["legal_cost"]["flops_used"]
    report = {"version": "v30.067.r22", "complete": True, "rows": rows,
              "before": "All14 sources, full1024columns, exactstructured contraction",
              "after": "Keep youngest8/4/1 sources; all other operands and algorithm frozen",
              "single_change": "source age cutoff only", "no_adjusted_score": True,
              "not_resource_acceptance": True,
              "selection_protocol": "Prespecified keep14/8/4/1; no reference/target access or model selection",
              "dataset_commit": meta["dataset_commit"], "network_id": meta["network_id"], "layer": 14,
              "capture_metadata_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "dependency_sha256": hashlib.sha256(Path("experiments/phase2/source_cap_structured.py").read_bytes()).hexdigest(),
              "limitations": ["Component bias is not final network MSE; cannot choose a submitted cutoff from this alone",
                              "FullK31/K211 diagrams and proxy removal unresolved"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
