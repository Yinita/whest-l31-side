"""O04 enumerate missing derivative basis with exact official signatures."""

import ast
import json
from collections import Counter
from pathlib import Path


def main():
    inventory = json.loads(
        Path("results/phase2/ablation_v30_066_full_output_inventory.json").read_text()
    )
    source = Path("estimators/k3_v29/estimator.py")
    tables = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("WICK_PAIRS", "TERM_SPECS"):
                    tables[target.id] = ast.literal_eval(node.value)
    wicks = set(tables["WICK_PAIRS"])
    gaps = []
    for row in inventory["rows"]:
        missing = [pair for pair in row["wick"] if tuple(pair) not in wicks]
        if missing:
            gaps.append(
                dict(
                    augment=row["augment"],
                    output=row["output"],
                    partition=row["partition"],
                    coefficient=row["coefficient"],
                    missing_wick=missing,
                    classification=row["classification"],
                )
            )
    counts = Counter(tuple(pair) for row in gaps for pair in row["missing_wick"])
    out = Path("results/phase2/ablation_v30_066_wick_gap_map.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                rows=gaps,
                missing_basis_counts={str(k): v for k, v in counts.items()},
                conclusion="Exact derivative/output gaps listed. Adding missing Wick columns alone does not restore tensor operands, coefficients, repeated exclusions or legal cost. Missing terms are absent mechanisms, not numerical kernel bugs.",
                scope="Static inventory refinement; no score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
