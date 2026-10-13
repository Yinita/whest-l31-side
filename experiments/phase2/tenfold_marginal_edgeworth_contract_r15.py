"""Independent quadrature versus the actual charged r15 marginal update."""
# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import json
import math
import textwrap
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import leggauss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    source = Path("estimators/tenfold_marginal_edgeworth2_r15/estimator.py")
    code = source.read_text().split("            # r15: fixed second-order Edgeworth closure", 1)[1]
    code = "            # r15: fixed second-order Edgeworth closure" + code.split(
        "            if not trim:\n                pk11 =", 1
    )[0]
    compiled = compile(textwrap.dedent(code), str(source), "exec")
    alpha = np.asarray([-4.0, -2.0, -0.2, 0.0, 0.7, 2.0, 4.0])
    sigma = np.asarray([0.3, 0.7, 1.0, 1.3, 0.5, 1.8, 2.0])
    phi = np.exp(-alpha**2 / 2) / math.sqrt(2 * math.pi)
    d3, k4 = 0.03 * sigma**3, -0.02 * sigma**4
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
        env = {"fnp": fnp, "mode": 1, "PK1": fnp.zeros((4, len(alpha))),
               "alpha": fnp.asarray(alpha), "phi": fnp.asarray(phi),
               "D3": fnp.asarray(d3), "g4row": fnp.asarray(k4),
               "ones_n": fnp.ones(len(alpha))}
        inv = fnp.asarray(1 / sigma)
        for degree in range(1, 7):
            env["inv" if degree == 1 else f"i{degree}"] = inv**degree
        exec(compiled, env)
        actual = np.asarray(env["PK1"]).copy()
    nodes, weights = leggauss(256)
    expected = np.zeros_like(actual)
    coefficient_checks = []
    for index, (a, s) in enumerate(zip(alpha, sigma, strict=True)):
        lo, hi = -a, 16.0
        z = lo + (nodes + 1) * (hi - lo) / 2
        density_weights = weights * (hi - lo) / 2 * np.exp(-z**2 / 2) / math.sqrt(2 * math.pi)
        for p in range(1, 5):
            for n, coefficient in ((6, d3[index]**2 / 72),
                                   (7, d3[index] * k4[index] / 144),
                                   (8, k4[index]**2 / 1152)):
                polynomial = [0.0] * n + [1.0]
                integrated = np.sum(density_weights * (s * (a + z))**p * hermeval(z, polynomial)) / s**n
                analytic = math.factorial(p) * s**(p - n) * phi[index] * hermeval(-a, [0.0] * (n - p - 1) + [1.0])
                coefficient_checks.append(abs(integrated - analytic))
                if p > 1 or n == 8:
                    expected[p - 1, index] += coefficient * integrated
    delta = float(np.max(np.abs(actual - expected)))
    # Cancellation makes the raw integrals least stable at sigma=.3. The
    # actual correction coefficient is the relevant production comparison.
    assert delta < 2e-13, delta
    report = {"version": "tenfold.marginal.edgeworth.contract.r15", "complete": True,
              "maximum_correction_abs_error": delta,
              "maximum_derivative_integral_abs_error": float(max(coefficient_checks)),
              "actual_correction": actual.tolist(), "expected_correction": expected.tolist(),
              "charged_cost": context.summary_dict(), "source": str(source),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Actual production block against independent 256point truncatedGaussian polynomial integration; fixed quadratic Edgeworth, no truth, no score/resource claim"}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "max_abs_error": delta,
                      "flops": report["charged_cost"]["flops_used"]}))


if __name__ == "__main__":
    main()
