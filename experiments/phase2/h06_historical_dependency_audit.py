"""Verify actual historical generator dependency definitions, not docstring counts."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path


def definitions(text):
    tree = ast.parse("\n".join(line for line in text.splitlines() if not line.startswith("type ")))
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def main():
    repo = "/tmp/whest-official-cumulant-reference"
    old = "6e80f7f2af0d33e252731ad9611dff17880b12fb"
    new = "93d091a4c26c042bfffa28f2e76a81bc0aba94bb"
    rows = []
    for file in ["partitions.py", "kprop_harmonic.py"]:
        texts = [
            subprocess.check_output(
                ["git", "-C", repo, "show", sha + ":src/mlp_kprop/" + file], text=True
            )
            for sha in (old, new)
        ]
        ds = [definitions(t) for t in texts]
        names = (
            set(ds[0])
            if file == "partitions.py"
            else {"get_int_cond", "get_vec_cond", "get_all_terms", "get_all_terms_iso"}
        )
        rows.append(
            dict(
                file=file,
                old_sha256=hashlib.sha256(texts[0].encode()).hexdigest(),
                new_sha256=hashlib.sha256(texts[1].encode()).hexdigest(),
                changed_definitions=sorted(
                    name for name in names if ds[0].get(name) != ds[1].get(name)
                ),
                missing_definitions=sorted(names - set(ds[1])),
            )
        )
    out = Path("results/phase2/ablation_v30_036_r05_historical_dependencies.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                old_commit=old,
                new_commit=new,
                rows=rows,
                scope="AST exact dependency audit, annotation changes included; changed definitions require actual historical extraction before asserting reproduction",
            ),
            indent=2,
        )
        + "\n"
    )
    print(out.read_text())


if __name__ == "__main__":
    main()
