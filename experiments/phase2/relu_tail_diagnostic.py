"""Offline comparison of frozen float32 Wick mean against high precision integral."""

import importlib.util
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.stats import norm


def main():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "frozen_v29", root / "estimators/k3_v29/estimator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = module.WICK_PAIRS.index((0, 1))
    const, exponent, p1, p2 = module.WICK_UNIFIED[index]
    mp.mp.dps = 70
    rows = []
    for sigma in [1e-6, 0.01, 1.0, 10.0]:
        for alpha in [-12, -8, -4, -1, 1, 4, 8, 12]:
            a = np.float32(alpha)
            powers = np.asarray([a**i for i in range(6)], dtype=np.float32)
            f = np.float32(norm.pdf(a))
            c = np.float32(norm.cdf(a))
            first = np.sum(powers[: len(p1)] * np.asarray(p1, dtype=np.float32), dtype=np.float32)
            second = np.sum(powers[: len(p2)] * np.asarray(p2, dtype=np.float32), dtype=np.float32)
            actual = float(
                np.float32(
                    (first * f + second * c) * np.float32(const) * np.float32(sigma) ** exponent
                )
            )
            x = mp.mpf(alpha)
            reference = mp.mpf(sigma) * mp.quad(
                lambda z: (z + x) * mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi), [-x, mp.inf]
            )
            rows.append(
                dict(
                    alpha=alpha,
                    sigma=sigma,
                    actual=actual,
                    reference=float(reference),
                    absolute_error=float(abs(mp.mpf(actual) - reference)),
                    relative_error=float(abs(mp.mpf(actual) - reference) / reference),
                    finite=bool(np.isfinite(actual)),
                )
            )
    for sigma in [1e-6, 0.01, 1.0, 10.0]:
        values = [r["actual"] for r in rows if r["sigma"] == sigma]
        assert all(a <= b for a, b in zip(values, values[1:]))
    output = root / "results/phase2/ablation_v30_125_relu_tail.json"
    assert not output.exists()
    output.write_text(
        json.dumps(
            dict(
                scope="Frozen Gaussian w(0,1) scalar float32 arithmetic; no high-order state or full GEMM path",
                rows=rows,
                monotonic=True,
                repair="None; diagnostic only",
            ),
            indent=2,
        )
        + "\n"
    )
    print("max relative error", max(r["relative_error"] for r in rows))


if __name__ == "__main__":
    main()
