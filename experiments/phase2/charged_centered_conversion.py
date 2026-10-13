"""Legally charged K31/K211 centering on exact finite-distribution diagnostic inputs."""

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
from probe_k211_conversion import cumulant


def main():
    output = Path("results/phase2/ablation_v30_067_r08_charged_centered_conversion.json")
    assert not output.exists()
    rng = np.random.default_rng(6708)
    rows = []
    for n in (4, 7):
        x = rng.normal(size=(n, 79))**2+rng.normal(size=(n, 1))
        probability = rng.uniform(size=79)
        probability /= probability.sum()
        mu = x@probability
        e2 = x*x@probability
        centered = x-mu[:, None]
        cov = (centered*probability)@centered.T
        p31, p21, exact31 = [np.zeros((n, n)) for _ in range(3)]
        p211, p111, exactcap = np.zeros((n, n, n)), np.zeros((n, n, n)), np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                p31[i, j] = cumulant([x[i]**3, x[j]], probability)
                p21[i, j] = cumulant([x[i]**2, x[j]], probability)
                if i != j:
                    exact31[i, j] = cumulant([x[i], x[i], x[i], x[j]], probability)
                for k in range(n):
                    p211[i, j, k] = cumulant([x[i]**2, x[j], x[k]], probability)
                    p111[i, j, k] = cumulant([x[i], x[j], x[k]], probability)
                    if len({i, j, k}) == 3:
                        exactcap[j, k] += cumulant([x[i], x[i], x[j], x[k]], probability)
        with flops.BudgetContext(flop_budget=2**35, quiet=True) as ctx:
            fm, fe2, fc, fp31, fp21, fp211, fp111 = map(
                fnp.asarray, (mu, e2, cov, p31, p21, p211, p111))
            k31 = fp31-3*fm[:, None]*fp21+(6*fm**2-3*fe2)[:, None]*fc
            fnp.fill_diagonal(k31, 0)
            # Contract the centered power terms, subtract repeated i=j/i=k,
            # then zero j=k. No dense candidate input assumption is hidden.
            corrected = fp211-2*fm[:, None, None]*fp111
            repeated_ab = fnp.stack([corrected[j, j, :] for j in range(n)])
            repeated_ac = fnp.stack([corrected[k, :, k] for k in range(n)], axis=1)
            diag = fnp.diag(fc)
            distinct_cov_product = fc.T@fc-diag[:, None]*fc-fc*diag[None, :]
            capped = fnp.sum(corrected, axis=0)-repeated_ab-repeated_ac-2*distinct_cov_product
            fnp.fill_diagonal(capped, 0)
        err31 = float(np.max(np.abs(np.asarray(k31)-exact31)))
        err211 = float(np.max(np.abs(np.asarray(capped)-exactcap)))
        assert max(err31, err211) < 1e-9
        rows.append({"width": n, "support_points": 79, "K31_max_error": err31,
                     "capped_K211_max_error": err211, "legal_cost": ctx.summary_dict(),
                     "input_values_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                     "probability_sha256": hashlib.sha256(probability.tobytes()).hexdigest()})
    report = {"version": "v30.067.r08", "complete": True, "rows": rows,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "partition_reference_sha256": hashlib.sha256(Path("experiments/phase2/probe_k211_conversion.py").read_bytes()).hexdigest(),
              "before": "Exact centering algebra only offline", "after": "Billed K31 and distinct capped K211 centering against independent partitions",
              "scope": "Float64 small-object centering contract; exact input power cumulants generated offline; no full production or competition candidate",
              "limitations": ["Dense K211/K111 inputs not feasible production representation at1024", "No actual V29 integration or score"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print([{k: v for k, v in row.items() if k != "legal_cost"} for row in rows])


if __name__ == "__main__":
    main()
