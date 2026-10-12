"""Read actual upstream lean generator body, compare fixed current official diagrams."""

import ast
import hashlib
import json
from pathlib import Path

from experiments.phase2.audit_official_augmented_terms import extract_generators


def main():
    output = Path("results/phase2/ablation_v30_066_r03_lean_generator_semantics.json")
    assert not output.exists()
    source = Path("/tmp/whest-p2-k3-reference/lean/lean_k3.py")
    ns, sources = extract_generators(Path("/tmp/whest-official-cumulant-reference"))
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "build_term_table"
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)
    table = ns["build_term_table"]()
    actual = {
        (ip, tuple(v for _, v in factors)): dict(coef=coef, wick=wick)
        for ip, factors, coef, wick in table
    }
    rows = []
    for ip, classes in ns["get_all_terms_iso"](3, d_max=4).items():
        for part, count in classes.items():
            dead = any(
                sum(v) == 4
                and tuple(sorted((x for x in v if x), reverse=True)) not in ((4,), (2, 2))
                for v in part
            )
            excluded = ip in ((3, 1), (2, 1, 1), (1, 1, 1)) or len(ip) > 3
            status = (
                "whole_output_excluded"
                if excluded
                else "assumed_zero_diagonal_metric_k4_slice"
                if dead
                else "retained_lean_full"
            )
            retained = (ip, part) in actual
            assert retained == (not excluded and not dead)
            rows.append(dict(output=ip, partition=part, classification=status, retained=retained))
    output.write_text(
        json.dumps(
            dict(
                lean_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                official_sources=sources,
                lean_full_terms=len(table),
                rows=rows,
                scope="Actual upstream build_term_table function extracted unchanged; regenerated with current frozen official generator",
                limitations=[
                    "Historical official dependency commit used by upstream unspecified; current-generation reproduction only",
                    "Rel1e-5 pruning file term_rels.json absent, cannot reconstruct exact historical pruning decisions",
                    "Diagonal-metric zero assumption does not apply to a general dense metric; full K31/K211 excluded deliberately",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print("Leanfull", len(table), "officialclasses", len(rows))


if __name__ == "__main__":
    main()
