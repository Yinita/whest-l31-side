"""Center capped K211 using factored K111, avoiding any charged n-cubed input."""

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


def center(cp_a, cp_b, cp_c, mu, cov, capped_power211):
    weighted_a_sum = fnp.sum(mu[:, None]*cp_a, axis=0)
    capped_mu111 = (cp_b*weighted_a_sum)@cp_c.T
    capped_mu111 = capped_mu111-(mu[:, None]*cp_a*cp_b)@cp_c.T
    capped_mu111 = capped_mu111-cp_b@(mu[:, None]*cp_a*cp_c).T
    diag = fnp.diag(cov)
    distinct_cov = cov.T@cov-diag[:, None]*cov-cov*diag[None, :]
    result = capped_power211-2*capped_mu111-2*distinct_cov
    fnp.fill_diagonal(result, 0)
    return result


def main():
    output = Path("results/phase2/ablation_v30_067_r09_cp_centered_k211.json")
    assert not output.exists()
    rng = np.random.default_rng(6709)
    rows = []
    for n in (4, 8, 32, 1024):
        a, b, c = [rng.normal(size=(n, 3)).astype(np.float32) for _ in range(3)]
        mu = rng.normal(size=n).astype(np.float32)
        raw = rng.normal(size=(n, n)).astype(np.float32)
        cov = (raw+raw.T)*.5
        power211 = rng.normal(size=(n, n)).astype(np.float32)
        af, bf, cf, mf, cv, pf = [x.astype(np.float64) for x in (a, b, c, mu, cov, power211)]
        if n <= 32:
            # Dense/offline expected-value oracle only; no charged cubic tensor.
            cube = np.einsum("ir,jr,kr->ijk", af, bf, cf)
            expected = np.zeros((n, n))
            for j in range(n):
                for k in range(n):
                    if j == k:
                        continue
                    distinct = [i for i in range(n) if i != j and i != k]
                    expected[j, k] = pf[j, k]-2*np.sum(mf[distinct]*cube[distinct, j, k])-2*np.sum(cv[distinct, j]*cv[distinct, k])
        else:
            # Independent direct distinct-index sums for 16 frozen entries.
            expected = None
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            result = center(*map(fnp.asarray, (a, b, c, mu, cov, power211)))
        got = np.asarray(result).astype(np.float64)
        if expected is not None:
            maximum = float(np.max(np.abs(got-expected)))
            relative = float(np.linalg.norm(got-expected)/np.linalg.norm(expected))
            assert relative < 2e-6
        else:
            maximum = 0.0
            for j, k in [(i*61, (i*67+5)%n) for i in range(16)]:
                ids = [i for i in range(n) if i != j and i != k]
                p111 = np.sum(af[ids]*bf[j]*cf[k], axis=1)
                entry = pf[j, k]-2*np.sum(mf[ids]*p111)-2*np.sum(cv[ids, j]*cv[ids, k])
                maximum = max(maximum, abs(got[j, k]-entry))
            relative = None
            assert maximum < 2e-3
        assert np.max(np.abs(np.diag(got))) == 0
        rows.append({"width": n, "rank": 3, "max_abs_error": maximum, "relative_error": relative,
                     "verification": "full explicit distinct-index oracle" if n <= 32 else "16 prespecified direct entries and full diagonal",
                     "largest_charged_array_elements": n*n, "legal_cost": ctx.summary_dict()})
    output.write_text(json.dumps({"version": "v30.067.r09", "complete": True, "rows": rows,
                                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "before": "Centering requires dense supplied K111 tensor",
                                 "after": "Exact distinct capped centering from rank3 CP K111 and covariance; no cubic charged input",
                                 "scope": "Component contract and costs only; arbitrary synthetic input factors, not actual V29 source mapping or score"}, indent=2, allow_nan=False)+"\n")
    print([{k: r[k] for k in ("width", "max_abs_error", "relative_error")} for r in rows])


if __name__ == "__main__":
    main()
