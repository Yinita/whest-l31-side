"""H06 static extraction of frozen lean SIMPLE production table."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from experiments.phase2.audit_official_augmented_terms import extract_generators


def main():
    repo = Path("/tmp/whest-official-cumulant-reference")
    ns, sources = extract_generators(repo)
    lean = Path("/tmp/whest-p2-k3-reference/lean/lean_k3_aug.py")
    tree = ast.parse(lean.read_text())
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_term_table"
    ]
    assert len(functions) == 1
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(lean), "exec"), ns)
    table = ns["build_term_table"]()
    reported_lean_docstring_count = 94
    old_sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "f4325dd^"], text=True
    ).strip()
    old_source = subprocess.check_output(
        ["git", "-C", str(repo), "show", old_sha + ":src/mlp_kprop/kprop_harmonic.py"], text=True
    )
    old_source = "\n".join(line for line in old_source.splitlines() if not line.startswith("type "))
    old_tree = ast.parse(old_source)
    old_ns = dict(ns)
    names = {"get_int_cond", "get_vec_cond", "get_all_terms", "get_all_terms_iso"}
    old_functions = [
        node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    future = ast.parse("from __future__ import annotations").body[0]
    exec(
        compile(
            ast.Module(body=[future] + old_functions, type_ignores=[]),
            "old_official_generator",
            "exec",
        ),
        old_ns,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(lean), "exec"), old_ns)
    old_table = old_ns["build_term_table"]()
    rows = [
        dict(output=out, factors=factors, coefficient=coef, wick=wick)
        for out, factors, coef, wick in table
    ]
    path = Path("results/phase2/ablation_v30_036_simple_table.json")
    path = path.with_name("ablation_v30_036_r02_generator_history.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                lean_sha256=hashlib.sha256(lean.read_bytes()).hexdigest(),
                official_sources=sources,
                generated_count=len(table),
                older_official_commit=old_sha,
                older_generated_count=len(old_table),
                older_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
                lean_docstring_count=reported_lean_docstring_count,
                rows=rows,
                conclusion="Frozen lean SIMPLE records generated with current frozen official generators; docstring count94 may be stale. This includes deliberate dead-slice/output filtering, not an unpruned complete restoration. Candidate mapping and one charged control pending.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(len(rows))


if __name__ == "__main__":
    main()
