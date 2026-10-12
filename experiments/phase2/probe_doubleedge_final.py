"""Independent full-tensor parity for a final-only doubled-edge K111 birth."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    rng = np.random.default_rng(1101)
    checks = []
    for n in (4, 7, 11):
        c = rng.normal(size=(n, n))
        c = (c + c.T) / 2
        np.fill_diagonal(c, 0)
        w1, w2, w3 = rng.normal(size=(3, n))
        a = w1[:, None] * c
        h = w2[:, None] * c**2 * w3[None, :]
        identity = np.eye(n)
        raw = np.zeros((n, n, n))
        for order in itertools.permutations(range(3)):
            fac = (a, identity, h)
            raw += 0.5 * np.einsum("ir,jr,kr->ijk", *(fac[i] for i in order))
        # Enumerate the diagram's six oriented paths independently of hub factors.
        diagram = np.zeros_like(raw)
        for i, j, k in itertools.product(range(n), repeat=3):
            coords = (i, j, k)
            for leaf_light, center, leaf_heavy in itertools.permutations(coords):
                diagram[i, j, k] += (
                    0.5
                    * w1[leaf_light]
                    * w3[center]
                    * w2[leaf_heavy]
                    * c[leaf_light, center]
                    * c[center, leaf_heavy] ** 2
                )
        r = a * h
        m = -3 * r.T
        corrected = raw.copy()
        for order in itertools.permutations(range(3)):
            fac = (m, identity, identity)
            corrected += np.einsum("ir,jr,kr->ijk", *(fac[i] for i in order)) / 6
        target = diagram.copy()
        for i, j, k in itertools.product(range(n), repeat=3):
            if len({i, j, k}) != 3:
                target[i, j, k] = 0
        W = rng.normal(size=(n, n))
        explicit_d3 = np.einsum("ai,aj,ak,ijk->a", W, W, W, target)
        aw, hw, mw = W @ a, W @ h, W @ m
        formula_d3 = 3 * np.sum(aw * W * hw, axis=1) + np.sum(mw * W**2, axis=1)
        raw_d21 = np.array([[raw[i, i, j] for j in range(n)] for i in range(n)])
        explicit_d21 = np.einsum("ai,aj,bk,ijk->ab", W, W, W, target)
        formula_d21 = (
            (aw * W) @ hw.T
            + (aw * hw) @ W.T
            + (W * hw) @ aw.T
            + (W**2) @ mw.T / 3
            + 2 * (mw * W) @ W.T / 3
        )
        errors = {
            "diagram_vs_sym_hub": float(np.max(np.abs(diagram - raw))),
            "birth_repeated_vs_AH": float(np.max(np.abs(raw_d21 - r))),
            "corrected_vs_zero_repeated": float(np.max(np.abs(corrected - target))),
            "final_d3": float(np.max(np.abs(formula_d3 - explicit_d3))),
            "final_d21": float(np.max(np.abs(formula_d21 - explicit_d21))),
        }
        assert max(errors.values()) < 5e-11, errors
        checks.append({"n": n, "errors": errors})
    candidate = Path("estimators/k3_v29_doubleedge_final/estimator.py")
    out = {
        "official_source": "https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py",
        "official_commit": "93d091a4c26c042bfffa28f2e76a81bc0aba94bb",
        "official_diagram": [[0, 1, 1], [0, 1, 1], [1, 0, 1]],
        "iso_count": 6,
        "vector_partition_coefficient": "1/2",
        "iso_aggregated_coefficient": "3",
        "formula": "zero_repeated(3 Sym(A,I,H)); A=diag(w1)Coff; H=diag(w2)(Coff**2)diag(w3); M=-3(A*H).T",
        "candidate": str(candidate),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "checks": checks,
        "cost": "Three n*n matmuls once per MLP plus O(n^2); n=1024: <=3*2^31/2^41 = 0.0029296875 budget before pointwise overhead.",
        "scope": "Only penultimate ReLU birth contributes to final D3. Prior layers and prior source histories unchanged; not full augmented K111 production.",
        "limitation": "Algebra-only finite-shape NumPy probe; no benchmark score or runtime resource claim.",
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"checks": checks, "candidate_sha256": out["candidate_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
