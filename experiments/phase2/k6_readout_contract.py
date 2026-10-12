"""O23 Hermite derivative sign/normalization for isolated sixth cumulant mean."""

import json
import math
from pathlib import Path


def main():
    rows = []
    for alpha in (-4, -1, 0, 0.5, 2, 4):
        for sigma in (0.5, 1, 2):
            phi = math.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi)
            # E[f^(6)(mu+sigmaZ)] = derivative_mu^4 Gaussian density at0.
            h4 = alpha**4 - 6 * alpha**2 + 3
            derivative = h4 * phi / sigma**5
            correction_coefficient = derivative / math.factorial(6)
            rows.append(
                dict(
                    alpha=alpha,
                    sigma=sigma,
                    w61=derivative,
                    kappa6_coefficient=correction_coefficient,
                )
            )
    path = Path("results/phase2/ablation_v30_085_k6_readout_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="deltaMean=kappa6*(alpha4-6alpha2+3)*phi/(720sigma5)",
                conclusion="Isolated linearκ6 ReLU mean coefficient derived from fourth derivative Gaussian density. This omits κ3², κ4² and other Edgeworth terms, and full validκ6 stateproduction is not available. No networkscore/candidate acceptance.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
