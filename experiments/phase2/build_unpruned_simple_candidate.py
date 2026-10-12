"""Reconstruct unpruned historical SIMPLE tables without term_rels thresholds."""

import ast
import hashlib
import json
import math
import pprint
import subprocess
from pathlib import Path

from experiments.phase2.audit_official_augmented_terms import extract_generators


def main():
    repo = Path("/tmp/whest-official-cumulant-reference")
    ns, sources = extract_generators(repo)
    old = "6e80f7f2af0d33e252731ad9611dff17880b12fb"
    text = subprocess.check_output(
        ["git", "-C", str(repo), "show", old + ":src/mlp_kprop/kprop_harmonic.py"], text=True
    )
    tree = ast.parse("\n".join(x for x in text.splitlines() if not x.startswith("type ")))
    names = {"get_int_cond", "get_vec_cond", "get_all_terms", "get_all_terms_iso"}
    defs = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    exec(
        compile(
            ast.Module(
                body=ast.parse("from __future__ import annotations").body + defs, type_ignores=[]
            ),
            "historical",
            "exec",
        ),
        ns,
    )
    lean = Path("/tmp/whest-p2-k3-reference/lean/lean_k3.py")
    defs = [
        x
        for x in ast.parse(lean.read_text()).body
        if isinstance(x, ast.FunctionDef) and x.name == "build_term_table"
    ]
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(lean), "exec"), ns)
    table = ns["build_term_table"]()
    parent = Path("estimators/k3_v29/estimator.py")
    static = {}
    for node in ast.parse(parent.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ["WICK_PAIRS", "WICK_UNIFIED", "TERM_SPECS"]
        ):
            static[node.targets[0].id] = ast.literal_eval(node.value)
    pairs = static["WICK_PAIRS"][:]
    unified = static["WICK_UNIFIED"][:]
    base2 = {
        None: "ones2",
        (1, 1): "c_off",
        (2, 1): "d21",
        (1, 2): "d21T",
        (2, 2): "wk4m",
        (3, 0): "d3col",
        (0, 3): "d3row",
        (4, 0): "g4col",
        (0, 4): "g4row",
    }
    base1 = {None: "ones1", (3,): "d3", (4,): "g4"}

    def index(kp):
        if kp not in pairs:
            k, p = kp
            assert k >= p, ("missing lower derivative polynomial", kp)
            m = k - p + 1
            if m == 1:
                u = (float(math.factorial(p)), 0, [], [1.0])
            else:
                he = [1.0]
                prev = [0.0]
                for degree in range(m - 2):
                    shifted = [0.0] + he
                    for j, v in enumerate(prev):
                        shifted[j] -= degree * v
                    prev, he = he, shifted
                u = (float(math.factorial(p)), -(m - 1), [(-1.0) ** (m - 2) * v for v in he], [])
            pairs.append(kp)
            unified.append(u)
        return pairs.index(kp)

    specs = {}
    for out, factors, coef, wick in table:
        base = base2 if len(out) == 2 else base1
        fs = [v for _, v in factors]
        assert len(fs) <= 2
        specs.setdefault(out, []).append(
            (
                base[fs[0] if fs else None],
                base[fs[1] if len(fs) > 1 else None],
                index(wick[0]),
                index(wick[1]) if len(out) == 2 else None,
                coef,
            )
        )
    out = Path("estimators/ablation_v30_036_r03")
    out.mkdir(exist_ok=False)
    source = parent.read_text()
    begin = source.index("WICK_PAIRS =")
    end = source.index("PK2K_TABLE =", begin)
    source = (
        source[:begin]
        + "".join(
            name + " = " + pprint.pformat(value, width=100) + "\n\n"
            for name, value in [
                ("WICK_PAIRS", pairs),
                ("WICK_UNIFIED", unified),
                ("TERM_SPECS", specs),
            ]
        )
        + source[end:]
    )
    source = source.replace(
        "ALPHA_DEG = 5", "ALPHA_DEG = " + str(max(max(len(u[2]), len(u[3])) for u in unified) - 1)
    )
    source = source.replace(
        "SIG_EMIN, SIG_EMAX = -6, 4",
        "SIG_EMIN, SIG_EMAX = " + repr((min(u[1] for u in unified), max(u[1] for u in unified))),
    )
    (out / "estimator.py").write_text(source)
    report = Path("results/phase2/ablation_v30_036_r03_unpruned_manifest.json")
    report.write_text(
        json.dumps(
            dict(
                count=len(table),
                wick_pairs=pairs,
                upstream_official_commit=old,
                candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
                scope="Unpruned90records historicalgenerator candidate; historicaldependency alignment inferred not proven, currentaug wk431 append retained; independent coefficient/numerical gates pending",
            ),
            indent=2,
        )
        + "\n"
    )
    print(len(table), len(pairs))


if __name__ == "__main__":
    main()
