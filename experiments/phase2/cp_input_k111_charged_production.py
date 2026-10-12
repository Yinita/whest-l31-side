"""Full augmented K111 polynomial and capped distinct output, legal component."""

import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.audit_official_augmented_terms import coefficient, extract_generators


def main():
    path = Path("results/phase2/ablation_v30_067_r07_cp_k111_charged_component.json")
    assert not path.exists()
    ns, sources = extract_generators(Path("/tmp/whest-official-cumulant-reference"))
    classes = ns["get_all_terms_iso"](3, d_max=4, augment=True)[(1, 1, 1)]
    terms = []
    for part, count in classes.items():
        orbit = {
            tuple(sorted(tuple(v[i] for i in permutation) for v in part))
            for permutation in tuple(itertools.permutations(range(3)))
        }
        assert len(orbit) == count
        terms.extend((p, float(coefficient(p, 1))) for p in orbit)
    rng = np.random.default_rng(6703)
    rows = []
    for n in (4, 8, 16, 32):
        vectors = {v for part, _ in terms for v in part}
        factors = {
            v: rng.normal(size=(n,) * sum(x > 0 for x in v)).astype("float32") for v in vectors
        }
        cp_inputs = {}
        for v in vectors:
            if sum(x > 0 for x in v) == 3:
                cp = [rng.normal(size=(n, 3)).astype("float32") for _ in range(3)]
                cp_inputs[v] = cp
                factors[v] = np.einsum("ar,br,cr->abc", *cp)
        pairs = {(sum(v[i] for v in part), p) for part, _ in terms for i, p in enumerate((1, 1, 1))}
        wicks = {pair: rng.normal(size=n).astype("float32") for pair in pairs}
        expected = np.zeros((n, n))
        for a, b, c in itertools.product(range(n), repeat=3):
            if len({a, b, c}) != 3:
                continue
            for part, coefficient_value in terms:
                value = coefficient_value
                for v in part:
                    value *= float(factors[v][tuple(i for i, s in zip((a, b, c), v) if s)])
                for axis, power in enumerate((1, 1, 1)):
                    value *= float(wicks[(sum(v[axis] for v in part), power)][(a, b, c)[axis]])
                expected[b, c] += value
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            fs = {v: fnp.array(x) for v, x in factors.items() if v not in cp_inputs}
            cp_fs = {v: [fnp.array(x) for x in parts] for v, parts in cp_inputs.items()}
            ws = {k: fnp.array(x) for k, x in wicks.items()}
            cap = fnp.zeros((n, n), dtype=fnp.float32)
            slab = 4
            for part, coefficient_value in terms:
                for start in range(0, n, slab):
                    stop = min(start + slab, n)
                    value = fnp.ones((stop - start, n, n), dtype=fnp.float32) * coefficient_value
                    for v in part:
                        if v in cp_fs:
                            operand = fnp.einsum(
                                "ar,br,cr->abc", cp_fs[v][0][start:stop], cp_fs[v][1], cp_fs[v][2]
                            )
                        else:
                            operand = fs[v][start:stop] if v[0] else fs[v]
                        shape = (stop - start if v[0] else 1, n if v[1] else 1, n if v[2] else 1)
                        value = value * fnp.reshape(operand, shape)
                    for axis, power in enumerate((1, 1, 1)):
                        operand = ws[(sum(v[axis] for v in part), power)]
                        if axis == 0:
                            operand = operand[start:stop]
                        shape = [1, 1, 1]
                        shape[axis] = stop - start if axis == 0 else n
                        value = value * fnp.reshape(operand, shape)
                    contracted = fnp.sum(value, axis=0)
                    zero = fnp.zeros(n, dtype=fnp.float32)
                    repeated_ab = fnp.stack(
                        [value[b - start, b, :] if start <= b < stop else zero for b in range(n)],
                        axis=0,
                    )
                    repeated_ac = fnp.stack(
                        [value[c - start, :, c] if start <= c < stop else zero for c in range(n)],
                        axis=1,
                    )
                    cap = cap + contracted - repeated_ab - repeated_ac
            fnp.fill_diagonal(cap, 0)
        relative = float(np.linalg.norm(np.asarray(cap) - expected) / np.linalg.norm(expected))
        assert relative < 2e-6
        rows.append(
            dict(
                n=n,
                raw_terms=len(terms),
                relative_error=relative,
                flops=ctx.summary_dict()["flops_used"],
                maximum_tensor_elements=min(n, slab) * n**2,
            )
        )
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                sources=sources,
                scope="Full augmented K111 blocked polynomial with rank3CP three-index inputs; dense expansion used only for offline expected-value oracle",
                limitations=[
                    "Intermediate slab4 at1024 needs16MiB; 3index inputs supplied as rank3CP, no charged n3 input; actual V29 source-to-CP mapping and approximation error unresolved",
                    "No complete candidate or score",
                    "Full centered conversion and source integration pending",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()
