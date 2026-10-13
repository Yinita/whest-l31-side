"""All columns of one real B1 source, using the frozen actual output mean.

Only the K111 Wick/centering orbit is represented; this is not the entire
K211 production object. Compare mean proxies without using target labels.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from real_source_k211_contract import capped_sym_source, direct_entry


def main():
    output = Path("results/phase2/ablation_v30_067_r12_full_b1_k211.json")
    assert not output.exists()
    provenance = Path("results/phase2/ablation_v30_067_r11_full_b1_capture.json")
    metadata = json.loads(provenance.read_text())
    capture = Path(metadata["capture"])
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == metadata["capture_sha256"]
    x = np.load(capture)
    rows = []
    for n in (8, 32, 1024):
        # Retain all1024 source columns even for small neuron-subset contracts.
        a, b, c = [x[k][:n].copy() for k in ("a", "p", "c")]
        mu, sigma, alpha = [x[k][:n].copy() for k in ("pk1v", "sigma", "alpha")]
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            fa, fb, fc, fm, sd, al = map(fnp.asarray, (a, b, c, mu, sigma, alpha))
            gate = flops.stats.norm.cdf(al).astype(fnp.float32)
            density = flops.stats.norm.pdf(al).astype(fnp.float32)
            gaussian = sd*(density+al*gate)
            # w(1,2) is a Gaussian Wick coefficient, independent of pk1v.
            weight = 2*gaussian-2*fm*gate
            centered = capped_sym_source(fa, fb, fc, weight, gate)
        # A separate budget records the one-factor ablation alone.
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as proxy_ctx:
            proxy = capped_sym_source(fa, fb, fc, 2*gaussian-2*gaussian*gate, gate)
        af, bf, cf = [v.astype(np.float64) for v in (a, b, c)]
        gf = np.asarray(gate).astype(np.float64)
        wf = np.asarray(weight).astype(np.float64)
        got = np.asarray(centered).astype(np.float64)
        entries = list(itertools.product(range(n), repeat=2)) if n <= 32 else [
            (i*61, (i*67+5)%n) for i in range(16)]
        expected = np.array([direct_entry(af, bf, cf, wf, gf, j, k) for j, k in entries])
        actual = np.array([got[j, k] for j, k in entries])
        error = float(np.max(np.abs(actual-expected)))
        scale = max(float(np.max(np.abs(expected))), 1e-12)
        assert error < 5e-5*scale+1e-8, (n, error, scale)
        assert np.max(np.abs(np.diag(got))) == 0
        assert np.max(np.abs(got-got.T)) < 1e-8
        delta = got-np.asarray(proxy).astype(np.float64)
        rows.append({"width": n, "source_columns": a.shape[1],
                     "verification": "full scalar distinct-index oracle" if n <= 32 else "16 frozen scalar entries plus full symmetry/diagonal",
                     "max_abs_error": error, "relative_max_error": error/scale,
                     "component_norm": float(np.linalg.norm(got)),
                     "gaussian_proxy_component_norm": float(np.linalg.norm(np.asarray(proxy))),
                     "actual_vs_gaussian_mean_max_abs_delta": float(np.max(np.abs(mu-np.asarray(gaussian)))),
                     "actual_vs_gaussian_core_delta_norm": float(np.linalg.norm(delta)*6/(n+4)),
                     "projected_core_norm": float(np.linalg.norm(got)*6/(n+4)),
                     "actual_vs_gaussian_relative_component_delta": float(np.linalg.norm(delta)/max(np.linalg.norm(got), 1e-30)),
                     "largest_charged_array_elements": n*n,
                     "actual_mean_legal_cost": ctx.summary_dict(),
                     "gaussian_proxy_legal_cost": proxy_ctx.summary_dict()})
    function = Path("experiments/phase2/real_source_k211_contract.py")
    result = {"version": "v30.067.r12", "complete": True, "rows": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "component_source_sha256": hashlib.sha256(function.read_bytes()).hexdigest(),
              "capture_sha256": metadata["capture_sha256"],
              "capture_provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
              "before": "First16 real B1 columns with Gaussian output mean proxy",
              "after": "All1024 columns of the same frozen B1 source with actual pk1v; separate Gaussian-proxy mean ablation",
              "scope": "Target-free single-source K111 Wick/centering orbit, allcolumns component diagnostic, not candidate score",
              "limitations": ["Only one of14 source slots; no feedback/feed/M legs",
                              "Remaining K211 diagram orbits, full K31 and proxy subtraction pending",
                              "No full1024x16 candidate contract, strictRPC or resource acceptance"],
              "no_adjusted_score": True}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print([{k: r[k] for k in ("width", "max_abs_error", "projected_core_norm", "actual_vs_gaussian_relative_component_delta")}
           for r in rows])
    print("width1024 actualmean FLOPs", rows[-1]["actual_mean_legal_cost"]["flops_used"])


if __name__ == "__main__":
    main()
