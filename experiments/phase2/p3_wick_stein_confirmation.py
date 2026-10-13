"""Independent Stein/Hermite integration for every p3 Wick derivative k1..5."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def independent_integral(mean, sd, k, order):
    # Stein identity uses ReLU^3 itself, including the distributional k4/k5
    # derivatives implicitly. No derivative expression being tested is used.
    nodes, weights = np.polynomial.legendre.leggauss(order)
    lo = -mean/sd
    hi = max(14.0, lo+14.0)
    z = lo+(nodes+1)*(hi-lo)/2
    hermite = np.polynomial.hermite_e.hermeval(z, [0]*k+[1])
    density = np.exp(-z*z/2)/math.sqrt(2*math.pi)
    return float(np.sum(weights*(mean+sd*z)**3*hermite*density)*(hi-lo)/(2*sd**k))


def main():
    output = Path("results/phase2/ablation_v30_067_r07_p3_wick_stein.json")
    assert not output.exists()
    previous = Path("results/phase2/ablation_v30_067_r06_p3_wick_charged.json")
    frozen = json.loads(previous.read_text())
    mu, sigma = np.array(frozen["mu"], dtype=np.float32), np.array(frozen["sigma"], dtype=np.float32)
    with flops.BudgetContext(flop_budget=2**30, quiet=True) as ctx:
        m, s = fnp.asarray(mu), fnp.asarray(sigma)
        a = m/s
        phi = flops.stats.norm.pdf(a).astype(fnp.float32)
        cdf = flops.stats.norm.cdf(a).astype(fnp.float32)
        legal = [3*((m*m+s*s)*cdf+m*s*phi), 6*(s*phi+m*cdf),
                 6*cdf, 6*phi/s, -6*a*phi/(s*s)]
    actual = np.stack([np.asarray(x) for x in legal], axis=1).astype(np.float64)
    approximations = [np.array([[independent_integral(float(m), float(s), k, order) for k in range(1, 6)]
                               for m, s in zip(mu, sigma)]) for order in (320, 512)]
    delta = float(np.max(np.abs(approximations[1]-approximations[0])))
    error = float(np.max(np.abs(actual-approximations[1])))
    assert delta < 1e-9
    assert np.allclose(actual, approximations[1], rtol=2e-5, atol=3e-6)
    result = {"version": "v30.067-r07-p3-Stein", "complete": True,
              "before": "k3..5 analytic formula only", "after": "all k1..5 independent truncated Stein integrals, 320/512 point refinement",
              "mu": mu.tolist(), "sigma": sigma.tolist(), "actual": actual.tolist(),
              "independent_expected": approximations[1].tolist(), "max_abs_error": error,
              "refinement_max_abs_delta": delta, "legal_cost": ctx.summary_dict(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "frozen_previous_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
              "scope": "Independent p3 k1..5 Wick contract; not full centered cumulant conversion, source integration or benchmark score",
              "limitations": ["Finite 14-sigma integration is not a certified quadrature error bound", "Only frozen five mu/sigma cases"]}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(error, delta, result["legal_cost"]["flops_used"])


if __name__ == "__main__":
    main()
