"""Independent finite support cumulants versus official direct pK->K coefficients."""

import ast
import itertools
import json
import math
from functools import cache
from pathlib import Path

import numpy as np

from experiments.phase2.audit_official_augmented_terms import extract_generators


def partitions(items):
    if not items:
        yield ()
        return
    first, *rest = items
    for part in partitions(rest):
        yield ((first,),) + part
        for i in range(len(part)):
            yield part[:i] + ((first,) + part[i],) + part[i + 1 :]


def main():
    root = Path(__file__).resolve().parents[2]
    repo = Path("/tmp/whest-official-cumulant-reference")
    ns, _ = extract_generators(repo)
    tree = ast.parse(
        "\n".join(
            s
            for s in (repo / "src/mlp_kprop/cumulants.py").read_text().splitlines()
            if not s.startswith("type ")
        )
    )
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pK_to_K_coef")
    fn.decorator_list = []
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + [fn], type_ignores=[]
    )
    ast.fix_missing_locations(module)
    exec(compile(module, "official_coef", "exec"), ns)
    samples = np.random.default_rng(87).normal(size=(17, 3))
    probabilities = np.arange(1, 18, dtype=float)
    probabilities /= probabilities.sum()

    @cache
    def moment(labels):
        return float(probabilities @ np.prod(samples[:, labels], axis=1)) if labels else 1.0

    @cache
    def cumulant_groups(groups):
        total = 0.0
        for part in partitions(list(range(len(groups)))):
            product = 1.0
            for block in part:
                product *= moment(
                    tuple(sorted(itertools.chain.from_iterable(groups[i] for i in block)))
                )
            total += (-1) ** (len(part) - 1) * math.factorial(len(part) - 1) * product
        return total

    rows = []
    for labels in [
        (0,) * 5,
        (0, 0, 0, 1, 1),
        (0, 0, 1, 1, 2),
        (0,) * 6,
        (0, 0, 0, 1, 1, 1),
        (0, 0, 1, 1, 2, 2),
    ]:
        actual = 0.0
        for part in partitions(list(range(len(labels)))):
            vectors = []
            value = 1.0
            for block in part:
                groups = tuple(
                    tuple(labels[i] for i in block if labels[i] == axis) for axis in range(3)
                )
                groups = tuple(g for g in groups if g)
                value *= cumulant_groups(groups)
                vectors.append(tuple(sum(labels[i] == axis for i in block) for axis in range(3)))
            actual += ns["_pK_to_K_coef"](tuple(vectors)) * value
        expected = cumulant_groups(tuple((label,) for label in labels))
        rows.append(
            dict(
                labels=labels,
                direct_conversion=actual,
                independent_cumulant=expected,
                absolute_error=abs(actual - expected),
            )
        )
    assert max(r["absolute_error"] for r in rows) < 1e-9
    output = root / "results/phase2/ablation_v30_087_degree6_conversion.json"
    assert not output.exists()
    output.write_text(
        json.dumps(
            dict(
                scope="Finite support17 atoms,6 degree5/6 repeated-index slices; official coefficient function unchanged AST body",
                source_commit="93d091a4c26c042bfffa28f2e76a81bc0aba94bb",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    print("max error", max(r["absolute_error"] for r in rows))


if __name__ == "__main__":
    main()
