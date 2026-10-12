"""L29 quadrature convergence on fixed actual prestate subset, no reference scoring."""

import json
from pathlib import Path

import numpy as np
from positive_readout_network_screen import mean_relu


def main():
    rows = []
    rules = {n: np.polynomial.legendre.leggauss(n) for n in (48, 96, 192)}
    for i in range(8):
        f = np.load(f"benchmark/data/v29_features_v30_096/network_{i}.npz")["features"][-1]
        for neuron in (0, 127, 255, 511, 767, 1023):
            mu, sigma, gamma = (
                float(f[neuron, 1]),
                float(f[neuron, 3]),
                float(f[neuron, 9] / f[neuron, 3] ** 3),
            )
            means = {n: mean_relu(mu, sigma, gamma, *rules[n]) for n in rules}
            assert all(x is not None and np.isfinite(x) for x in means.values())
            rows.append(
                dict(
                    network_id=i,
                    neuron=neuron,
                    means=means,
                    delta96_192=abs(means[96] - means[192]),
                )
            )
    path = Path("results/phase2/ablation_v30_124_positive_quadrature_convergence.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                max_delta96_192=max(r["delta96_192"] for r in rows),
                conclusion="Fixed48neuron subset quadrature convergence measured without labels; cannot explain all8192neurons or validate family accuracy. Original8network negative score preserved.",
                scope="Numerical integration diagnostic",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
