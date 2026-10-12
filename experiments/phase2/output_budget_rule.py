"""L26 fixedreference-free proxy topk budget rule, no rank sweep."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(126)
    rows = []
    for count in (7, 11, 19):
        contribution = rng.normal(size=(count, 23))
        scores = np.linalg.norm(contribution, axis=1)
        slots = 2
        selected = sorted(range(count), key=lambda j: (-scores[j], j))[:slots]
        assert len(set(selected)) == slots
        # Squarednormsum of omitted contributions is optimal for this surrogate only.
        selected_energy = float(np.sum(scores[selected] ** 2))
        competitor = max(
            sum(scores[list(s)] ** 2)
            for s in __import__("itertools").combinations(range(count), slots)
        )
        assert abs(selected_energy - competitor) < 1e-10
        rows.append(
            dict(
                source_count=count,
                frozen_dense_slots=slots,
                selected=selected,
                selected_proxy_energy=selected_energy,
            )
        )
    path = Path("results/phase2/ablation_v30_121_output_budget_rule.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Fixed2slot top proxy-norm selection deterministic and optimizes additive surrogateenergy. Correlatedsource cancellation and nonlinearfeedback may invalidate surrogate; liveV29 sourceproxy, completeeviction/repeatcontract and actualscore required. No oraclelabels used.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
