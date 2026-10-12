"""Power counting of every algebraic product in lean X1Y1/X2Y2 birth."""

import hashlib
import itertools
import json
from pathlib import Path


def main():
    source = Path("/tmp/whest-p2-k3-reference/lean/lean_k3_aug.py")
    terms = {
        "B1_X": [("C", (1, 1, 0), "3*w1_i"), ("K21", (2, 1, 0), "3*w2_i/2")],
        "B1_Y": [
            ("C", (0, 1, 1), "w2_j*w1_k"),
            ("K21", (0, 2, 1), "w3_j*w1_k/2"),
            ("K12", (0, 1, 2), "w2_j*w2_k/2"),
            ("K22", (0, 2, 2), "w3_j*w2_k/4"),
        ],
        "B2_X": [("K12", (1, 2, 0), "3*w1_i/2"), ("K22", (2, 2, 0), "3*w2_i/4")],
        "B2_Y": [
            ("C", (0, 1, 1), "w3_j*w1_k"),
            ("K21", (0, 2, 1), "w4_j*w1_k/2"),
            ("K12", (0, 1, 2), "w3_j*w2_k/2"),
            ("K22", (0, 2, 2), "w4_j*w2_k/4"),
        ],
    }
    rows = []
    for block in ["B1", "B2"]:
        for left, right in itertools.product(terms[block + "_X"], terms[block + "_Y"]):
            vectors = [left[1], right[1]]
            cost = sum(sum(v) - 1 - int(all(x % 2 == 0 for x in v)) for v in vectors)
            rows.append(
                dict(
                    block=block,
                    left=left[0],
                    right=right[0],
                    vectors=vectors,
                    wick_factors=[left[2], right[2]],
                    power_cost=cost,
                    base_budget=cost <= 2,
                    augmented_budget=cost <= 3,
                    implemented_v29="B1 C*C and two C*K21 feedback paths plus feedback*feedback only; B2 shadowed; WK22 product paths omitted",
                )
            )
    out = Path("results/phase2/ablation_v30_080_birth_power_inventory.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                scope="All16 X1Y1/X2Y2 expansion products, frozen public lean formula; no restored candidate",
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                rows=rows,
                conclusion="Composite feedback*feedback has cost4, exceeds augmented3; do not call mixed birth strict budgetK3. Requires coefficient/isomorphism matching before restoration",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
