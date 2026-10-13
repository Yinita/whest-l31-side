"""Actual lower-support K211 minus CovCov diagnostic; source centering omitted."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.real_k211_matrix_production import contract_diagram, matrix_wicks, table


def main():
    output = Path("results/phase2/ablation_v30_067_r31_actual_covcov_gate.json")
    assert not output.exists()
    spec_path = Path("results/phase2/ablation_v30_067_r30_real_k211_matrix.json")
    specification = json.loads(spec_path.read_text())
    terms = [(tuple(tuple(v) for v in part), coef) for part, coef in specification["retained_raw_terms"]]
    rows = []
    for rversion in (24, 27):
        meta = json.loads(Path(f"results/phase2/ablation_v30_067_r{rversion}_real_k31_capture.json").read_text())
        capture = Path(meta["capture"])
        assert hashlib.sha256(capture.read_bytes()).hexdigest() == meta["capture_sha256"]
        with np.load(capture) as src:
            full = {k: src[k].copy() for k in ("mu", "sigma", "C_off", "D21", "wk4m", "pk11")}
        for n in (8, 32, 1024):
            data = {k: x[:n, :n] if x.ndim == 2 else x[:n] for k, x in full.items()}
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                fs = {v: fnp.asarray(x).astype(fnp.float64) for v, x in table(data).items()}
                ws = matrix_wicks(fnp.asarray(data["mu"]).astype(fnp.float64),
                                  fnp.asarray(data["sigma"]).astype(fnp.float64))
                power = fnp.zeros((n, n), dtype=fnp.float64)
                for part, coef in terms:
                    power = power+contract_diagram(part, coef, fs, ws, n)
                power = (power+power.T)/2
                prior = ctx.summary_dict()["flops_used"]
                cov = fnp.asarray(data["pk11"]).astype(fnp.float64)
                # Rows of captured pk11 identify the repeated-index coordinate.
                # Include subtraction even if captured diagonal happens to be 0.
                diagonal = fnp.diag(cov)
                capped_covcov = cov.T@cov-diagonal[:, None]*cov-cov.T*diagonal[None, :]
                fnp.fill_diagonal(capped_covcov, 0)
                partial = power-2*capped_covcov
                covcov_flops = ctx.summary_dict()["flops_used"]-prior
            c = data["pk11"].astype(np.float64)
            entries = list(itertools.product(range(n), repeat=2)) if n <= 32 else [
                (i*61, (i*67+5)%n) for i in range(16)]
            expected = [sum(c[i, j]*c[i, k] for i in range(n) if i != j and i != k)
                        if j != k else 0 for j, k in entries]
            got = np.array([np.asarray(capped_covcov)[j, k] for j, k in entries])
            error = float(np.max(np.abs(got-expected)))
            assert error < 1e-12
            p, q, delta = map(np.asarray, (power, partial, capped_covcov))
            rows.append({"network_id": meta["network_id"], "width": n,
                         "covcov_scalar_max_abs_error": error, "covcov_component_flops": covcov_flops,
                         "lower_support_power_norm": float(np.linalg.norm(p)),
                         "twice_covcov_norm": float(np.linalg.norm(2*delta)),
                         "partial_difference_norm": float(np.linalg.norm(q)),
                         "difference_vs_power_norm_ratio": float(np.linalg.norm(q)/np.linalg.norm(p)),
                         "legal_cost": ctx.summary_dict(),
                         "not_full_centered_K211": True})
            print({k:v for k,v in rows[-1].items() if k!='legal_cost'}, flush=True)
    output.write_text(json.dumps({"version": "v30.067.r31", "complete": True, "rows": rows,
                                 "before": "Actual lower-support K211 raw power component only",
                                 "after": "Subtract exact actual output-covariance CovCov contribution",
                                 "scope": "Partial cancellation diagnostic; three-index orbits and -2mean*pK111 still absent",
                                 "no_adjusted_score": True, "not_resource_acceptance": True,
                                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "specification_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                                 "dependency_sha256": hashlib.sha256(Path('experiments/phase2/real_k211_matrix_production.py').read_bytes()).hexdigest(),
                                 "formula": "CovCov_jk=sum_{i distinct j,k} cov_ij cov_ik, only j!=k",
                                 "next": "Complete pK111 birth/tree centering and three-index K4 terms, then proxy dedup"},
                                indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
