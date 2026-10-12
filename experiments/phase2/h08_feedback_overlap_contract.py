"""B1 WK12 term versus actual V29 Yt degree and factor orientation."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(3802)
    rows = []
    for n, r in ((4, 2), (7, 3), (11, 4)):
        q, b = rng.normal(size=(2, n, r))
        d = q @ b.T
        w1, w2, w3 = rng.normal(size=(3, n))
        wk12 = .5 * d
        desired = (w2[:, None] * w2[None, :] * wk12).T
        actual = (w1[:, None] * b) @ (q * (0.5 * w3)[:, None]).T
        wrong_substitute = (w2[:, None] * b) @ (q * (0.5 * w2)[:, None]).T
        assert np.allclose(desired, wrong_substitute)
        rows.append(
            dict(
                n=n,
                rank=r,
                desired_y_rms=float(np.sqrt(np.mean(desired**2))),
                v29_y_rms=float(np.sqrt(np.mean(actual**2))),
                difference_rms=float(np.sqrt(np.mean((desired - actual) ** 2))),
                thin_formula_error=float(np.max(abs(desired - wrong_substitute))),
            )
        )
    out = Path("results/phase2/ablation_v30_038_r02_feedback_overlap.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                rows=rows,
                source="V29 F2=w1*Bm.T,R2=.5*Qf*w3; exactWK12 target F2=w2*Bm.T,R2=.5*Qf*w2",
                conclusion="ExistingYt covers WK21-oriented w1/w3 derivative term, not WK12 w2/w2 term. Do not replace existingYt; append separate symmetricCP contribution and subtract exact birth repeats. Synthetic lowrank object only; sourceintegration/score remains",
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()
