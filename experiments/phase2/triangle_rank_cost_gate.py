"""H32 cost growth of exact CP expansion, not a bounded-rank claim."""

import json
from pathlib import Path


def main():
    n = 1024
    budget = 2**41
    rows = []
    for rank in (1, 2, 4, 8, 16, 32):
        columns = n * rank * rank
        transport = 6 * n * n * columns
        d21 = 6 * n * n * columns
        rows.append(
            dict(
                edge_rank=rank,
                cp_columns=columns,
                transport_flops=transport,
                d21_flops=d21,
                budget_fraction=(transport + d21) / budget,
            )
        )
    p = Path("results/phase2/ablation_v30_062_cp_cost_gate.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                formula="Direct tree CP from dense hub and rankR edge requires nR² columns; three transports and three D21 products each6n²columns leading FLOPs",
                conclusion="Exact CP is not rankR²: even rank16 exceeds full network budget before other work. Structured contractions or approximate compression necessary; these counts exclude all copies/QR and are lower operation-count projections.",
                scope="Analytic cost rejection of direct representation, no score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
