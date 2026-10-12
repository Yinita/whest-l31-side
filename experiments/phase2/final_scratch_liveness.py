"""C28 static scratch use bounds; explicit dynamic-alias review required."""

import ast
import hashlib
import json
from pathlib import Path


def main():
    source = Path("estimators/k3_v29/estimator.py")
    tree = ast.parse(source.read_text())
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_predict_core", "_dslices", "_sym_product", "_hub2")
    ]
    rows = []
    for fn in functions:
        uses = {}
        allocations = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Name):
                uses.setdefault(node.id, []).append(node.lineno)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "get_pair")
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    allocations.append(dict(pool_key=node.args[0].value, line=node.lineno))
        names = [
            "T1",
            "T2",
            "abbuf",
            "PK2",
            "ABstack",
            "legs",
            "legs4",
            "bufs",
            "A_st",
            "P_st",
            "Z_st",
            "Zf_st",
        ]
        rows.append(
            dict(
                function=fn.name,
                scratch_name_bounds={
                    k: {"first": min(uses[k]), "last": max(uses[k])} for k in names if k in uses
                },
                pool_allocations=allocations,
            )
        )
    path = Path("results/phase2/ablation_v30_028_static_liveness.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                rows=rows,
                conclusion="Static lexical use bounds only: function calls, pool references, slices and inter-layer views escape localnames. Cannot infer freeable storage from last lexical use. Allocation-poison dynamic oracle and finalpath alias proof pending; no scratch reuse candidate accepted.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
