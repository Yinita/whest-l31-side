"""O04 exact bounded full-output inventory with explicit representability labels."""

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.phase2.audit_official_augmented_terms import (
    coefficient,
    extract_generators,
    factor_info,
)


def main():
    repo = Path("/tmp/whest-official-cumulant-reference")
    ns, sources = extract_generators(repo)
    source = Path("estimators/k3_v29/estimator.py")
    tree = ast.parse(source.read_text())
    tables = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("TERM_SPECS", "WICK_PAIRS"):
                    tables[target.id] = ast.literal_eval(node.value)
    wicks = set(tables["WICK_PAIRS"])
    rows = []
    for augment in (False, True):
        for output, terms in ns["get_all_terms_iso"](3, d_max=4, augment=augment).items():
            for partition, count in terms.items():
                factors = [factor_info(v) for v in partition]
                derivatives = ns["check_vec_partition"](partition, len(output))
                available_wicks = all((a, b) in wicks for a, b in zip(derivatives, output))
                kept = ns["factored_keeps_term"](3, output, partition)
                # Classification never equates abstract representation with numeric parity.
                if not kept:
                    status = "official_factored_pruned"
                elif len(output) > 2:
                    status = "factored_representation_requires_individual_audit"
                elif not available_wicks:
                    status = "missing_wick_basis"
                elif all(f["available_dense_v29"] for f in factors):
                    status = "dense_representable_needs_coefficient_map"
                else:
                    status = "missing_dense_operand"
                rows.append(
                    dict(
                        augment=augment,
                        output=output,
                        partition=partition,
                        coefficient=str(coefficient(partition, count)),
                        factors=factors,
                        wick=list(zip(derivatives, output)),
                        classification=status,
                    )
                )
    out = dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        official_sources=sources,
        rows=rows,
        counts=dict(Counter(r["classification"] for r in rows)),
        conclusion="All official bounded source/output signatures enumerated; labels expose pruning, representation and coefficient-map gates. Existing pair19/22 overlap remains separate evidence; complete V29 semantic equivalence is not established.",
        adjusted_score=None,
    )
    path = Path("results/phase2/ablation_v30_066_full_output_inventory.json")
    assert not path.exists()
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(len(rows), out["counts"])


if __name__ == "__main__":
    main()
