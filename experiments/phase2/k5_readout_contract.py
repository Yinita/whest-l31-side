"""O20 isolated fifth cumulant ReLU coefficient, no unjustified closure."""

import json
import math
from pathlib import Path


def main():
    rows = []
    for alpha in (-4, -1, 0, 0.5, 2, 4):
        for sigma in (0.5, 1, 2):
            phi = math.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi)
            derivative = -(alpha**3 - 3 * alpha) * phi / sigma**4
            rows.append(
                dict(alpha=alpha, sigma=sigma, w51=derivative, kappa5_coefficient=derivative / 120)
            )
    path = Path("results/phase2/ablation_v30_082_k5_readout_contract.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="deltaMean=−kappa5*(alpha3−3alpha)*phi/(120sigma4)",
                conclusion="Isolated linearκ5 mean coefficient analytic; at same derivativeorder κ3κ4 and other terms remain separate. Marginalbirthκ5 is not full propagatedclosure; validstate and networkscore pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
