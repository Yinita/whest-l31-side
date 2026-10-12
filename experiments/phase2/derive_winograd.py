"""Independent polynomial proof of seven-product Winograd schedule."""

import json
from pathlib import Path

import numpy as np


def winograd(a, b):
    a11, a12, a21, a22 = a
    b11, b12, b21, b22 = b
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    p1 = a11 * b11
    p2 = a12 * b21
    p3 = s4 * b22
    p4 = a22 * t4
    p5 = s1 * t1
    p6 = s2 * t2
    p7 = s3 * t3
    u1 = p1 + p2
    u2 = p1 + p6
    u3 = u2 + p7
    u4 = u2 + p5
    return u1, u4 + p3, u3 - p4, u3 + p5


def main():
    # Basis products span every bilinear polynomial coefficient independently.
    errors = []
    for i in range(4):
        for j in range(4):
            a = np.eye(4, dtype=int)[i]
            b = np.eye(4, dtype=int)[j]
            expected = (
                a[0] * b[0] + a[1] * b[2],
                a[0] * b[1] + a[1] * b[3],
                a[2] * b[0] + a[3] * b[2],
                a[2] * b[1] + a[3] * b[3],
            )
            errors.append(int(np.max(abs(np.asarray(winograd(a, b)) - expected))))
    assert max(errors) == 0
    p = Path("results/phase2/ablation_v30_002_winograd_derivation.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                scope="Exact bilinear coefficient proof only, plain mm schedule; no implementation or score yet",
                basis_cases=16,
                max_error=max(errors),
                products=7,
                additions=15,
                baseline_additions=18,
                dependency="Need pooled alias/liveness implementation; original hub must retain original schedule",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
