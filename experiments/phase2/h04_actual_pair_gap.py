"""Fixed actual state Gaussian pair truncation diagnostic; no targets."""

import json
import math
from pathlib import Path

import numpy as np


def ndtr(x):
    return np.vectorize(lambda t: 0.5 * math.erfc(-float(t) / math.sqrt(2)))(x)


def main():
    d = dict(np.load("benchmark/data/v29_real_source_v30_059_r05/network_0.npz"))
    c, sigma, alpha = d["covariance"], d["sigma"], d["alpha"]
    rho = c / sigma[:, None] / sigma[None, :]
    mask = ~np.eye(len(alpha), dtype=bool)
    rng = np.random.default_rng(3404)
    pairs = rng.integers(0, len(alpha), (256, 2))
    # Include the most correlated pair to disclose the tail, not only random pairs.
    extreme = np.unravel_index(np.argmax(np.where(mask, abs(rho), -1)), rho.shape)
    pairs = np.vstack([pairs, extreme])
    rows = []
    for order in (128, 256):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        errors = {key: [] for key in ("11", "21", "22")}
        for i, j in pairs:
            if i == j:
                continue
            a, b, r = float(alpha[i]), float(alpha[j]), float(rho[i, j])
            lo = max(-a, -12)
            x = (nodes + 1) * (12 - lo) / 2 + lo
            z = b + r * x
            sd = math.sqrt(max(1 - r * r, 1e-30))
            phi = np.exp(-((z / sd) ** 2) / 2) / math.sqrt(2 * math.pi)
            gate = ndtr(z / sd)
            y1 = sd * phi + z * gate
            y2 = (z * z + sd * sd) * gate + z * sd * phi
            density = np.exp(-x * x / 2) / math.sqrt(2 * math.pi) * weights * (12 - lo) / 2

            def marginal(t):
                p = ndtr(t)
                f = math.exp(-t * t / 2) / math.sqrt(2 * math.pi)
                m = f + t * p
                return [m, p, f], [(t * t + 1) * p + t * f, 2 * m, 2 * p]

            aa, aa2 = marginal(a)
            bb, bb2 = marginal(b)
            for key, left, right, y, power in [
                ("11", aa, bb, y1, 1),
                ("21", aa2, bb, y1, 2),
                ("22", aa2, bb2, y2, 2),
            ]:
                exact = float(np.sum((x + a) ** power * y * density))
                truncated = (
                    left[0] * right[0] + r * left[1] * right[1] + r * r * left[2] * right[2] / 2
                )
                errors[key].append(exact - truncated)
        rows.append(dict(order=order, gaps=errors))
    report = dict(
        network=0,
        layer=14,
        seed=3404,
        pairs=len(rows[0]["gaps"]["11"]),
        rho_quantiles=np.quantile(abs(rho[mask]), [0.5, 0.9, 0.99, 0.999, 1]).tolist(),
        gaps={
            key: dict(
                rms=float(np.sqrt(np.mean(np.array(rows[1]["gaps"][key]) ** 2))),
                max_abs=float(np.max(abs(np.array(rows[1]["gaps"][key])))),
                quadrature_delta_max=float(
                    np.max(abs(np.array(rows[1]["gaps"][key]) - rows[0]["gaps"][key]))
                ),
            )
            for key in ("11", "21", "22")
        },
        scope="Actual prestate standardized pureGaussian pK11/pK21/pK22, deterministic256pairs+maxrho; excludes mixed cumulants and statefeedback; no score",
        score_before=None,
        score_after=None,
    )
    out = Path("results/phase2/ablation_v30_034_r02_actual_pair_gap.json")
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
