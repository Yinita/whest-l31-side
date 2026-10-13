"""Independent small-tensor check of the V29 feed symmetry precondition."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from real_source_full_k211 import independent_v29_slices


def check(values):
    a, p, z, ell, w, s, e, c1, c2, y, f1, f2, r1, r2 = values
    n = len(a)
    blocks = [(3*a+f1@r1.T, p, a*w+f2@r2.T),
              (p*s+3*a*e+z@ell.T, p, p),
              (a*c1+p*c2, p, np.broadcast_to(y[:, None], p.shape))]
    dense = np.zeros((n, n, n))
    for x, b, c in blocks:
        raw = np.einsum("ir,jr,kr->ijk", x, b, c)
        dense += sum(raw.transpose(perm) for perm in itertools.permutations(range(3)))/6
    d3, d21 = independent_v29_slices(values)
    expected21 = np.array([[dense[i, i, j] if i != j else 0 for j in range(n)] for i in range(n)])
    feed = (a*c1+p*c2)@p.T
    return {"D3_max_abs_error": float(np.max(np.abs(d3-np.einsum("iii->i", dense)))),
            "D21_max_abs_error": float(np.max(np.abs(d21-expected21))),
            "feed_asymmetry": float(np.max(np.abs(feed-feed.T)))}


def main():
    output = Path("results/phase2/ablation_v30_067_r16_feed_symmetry_contract.json")
    assert not output.exists()
    rng = np.random.default_rng(6716)
    rows = []
    for n in (4, 7, 12):
        c1 = rng.uniform(.4, 1.4, size=n)
        raw = rng.normal(size=(n, n))
        covariance = (raw+raw.T)/2
        a, p = covariance/c1[None, :], np.eye(n)
        z, ell = [rng.normal(size=(n, 3)) for _ in range(2)]
        w, s, e, c2, y = [rng.normal(size=n) for _ in range(5)]
        f1, f2, r1, r2 = [rng.normal(size=(n, 2)) for _ in range(4)]
        values = [a, p, z, ell, w, s, e, c1, c2, y, f1, f2, r1, r2]
        birth = check(values)
        transport = rng.normal(size=(n, n))
        mapped = [transport@x if i in (0, 1, 2, 9, 10, 11) else x for i, x in enumerate(values)]
        carried = check(mapped)
        arbitrary = list(mapped)
        arbitrary[0] = arbitrary[0]+rng.normal(size=(n, n))
        broken = check(arbitrary)
        assert max(birth.values()) < 1e-10
        assert max(carried.values()) < 1e-9
        assert broken["D21_max_abs_error"] > 1e-4
        rows.append({"width": n, "lawful_birth": birth, "lawful_transported": carried,
                     "arbitrary_nonlawful_counterexample": broken})
    output.write_text(json.dumps({"version": "v30.067.r16", "complete": True, "rows": rows,
                                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  "slice_equation_source_sha256": hashlib.sha256(Path("experiments/phase2/real_source_full_k211.py").read_bytes()).hexdigest(),
                                  "before": "Arbitrary source operands gave a D21 mapping discrepancy",
                                  "after": "Full feed slice mapping requires X3 P^T symmetry; lawful birth/shared transport passes",
                                  "scope": "Offline independent float64 fulltensor contract/counterexample; no legal cost or score",
                                  "conclusion": "Keep the production-state symmetry precondition explicit; arbitrary synthetic failure is not a V29 bug",
                                  "no_adjusted_score": True}, indent=2, allow_nan=False)+"\n")
    print(rows)


if __name__ == "__main__":
    main()
