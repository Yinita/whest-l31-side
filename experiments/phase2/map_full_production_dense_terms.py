"""Map official full dense output classes to actual V29 tables without proxy equivalence."""

import ast
import hashlib
import json
import math
from pathlib import Path

from experiments.phase2.audit_official_augmented_terms import coefficient, extract_generators


def main():
    output = Path("results/phase2/ablation_v30_066_r02_dense_production_map.json")
    assert not output.exists()
    source = Path("estimators/k3_v29/estimator.py")
    nodes = ast.parse(source.read_text()).body
    selected = []
    for node in nodes:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("WICK_PAIRS", "TERM_SPECS", "_I")
            for t in node.targets
        ):
            selected.append(node)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "append"
            and ast.unparse(node).startswith("TERM_SPECS[")
        ):
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    wicks = namespace["WICK_PAIRS"]

    def signature(names, pairs):
        return tuple(sorted(names)), tuple(tuple(p) for p in pairs)

    actual = {}
    for ip, terms in namespace["TERM_SPECS"].items():
        if len(ip) != 2:
            continue
        actual[ip] = {}
        for a, b, wl, wr, c in terms:
            if wr is not None:
                key = signature((a, b), (wicks[wl], wicks[wr]))
                actual[ip][key] = actual[ip].get(key, 0) + c
    names = {
        (1, 1): "c_off",
        (0, 3): "d3row",
        (3, 0): "d3col",
        (0, 4): "g4row",
        (4, 0): "g4col",
        (2, 1): "d21",
        (1, 2): "d21T",
        (2, 2): "wk4m",
        (3, 1): "wk431",
        (1, 3): "wk431",
    }
    ns, sources = extract_generators(Path("/tmp/whest-official-cumulant-reference"))
    rows = []
    for augment in (False, True):
        official = ns["get_all_terms_iso"](3, d_max=4, augment=augment)
        for ip, classes in official.items():
            if len(ip) != 2:
                continue
            for partition, count in classes.items():
                operands = [names.get(v, "unmapped") for v in partition]
                if len(operands) == 1:
                    operands.append("ones2")
                pairs = list(zip(ns["check_vec_partition"](partition, len(ip)), ip))
                expected = float(coefficient(partition, count))
                existing = (
                    actual.get(ip, {}).get(signature(operands, pairs))
                    if len(operands) == 2
                    else None
                )
                classification = "missing_table_term"
                if len(operands) > 2:
                    classification = "requires_explicit_product_operand"
                elif "unmapped" in operands:
                    classification = "unmapped_operand"
                elif existing is not None:
                    classification = (
                        "table_match_proxy_operand"
                        if math.isclose(existing, expected, rel_tol=1e-12)
                        else "coefficient_mismatch_requires_orbit_audit"
                    )
                rows.append(
                    dict(
                        augment=augment,
                        output=ip,
                        partition=partition,
                        operands=operands,
                        wick=pairs,
                        official_coefficient=expected,
                        v29_coefficient=existing,
                        classification=classification,
                    )
                )
    output.write_text(
        json.dumps(
            dict(
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                official_sources=sources,
                rows=rows,
                scope="Static dense-output table audit, includes dynamic append statements. A table match is not proof that harmonic proxy tensors equal exact cumulants; full K211/K111 contractions remain separately unresolved",
                score=None,
            ),
            indent=2,
        )
        + "\n"
    )
    print("Mapped", len(rows), "dense classes")


if __name__ == "__main__":
    main()
