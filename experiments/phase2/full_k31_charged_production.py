"""Exact full diagram K31 dense production component using charged operands."""

import json
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.audit_official_augmented_terms import coefficient, extract_generators


def ndtr(x):
    return np.array([0.5 * (1 + math.erf(float(v) / math.sqrt(2))) for v in x])


def wick(k, p, mu, sigma):
    alpha = mu / sigma
    phi = np.exp(-(alpha**2) / 2) / np.sqrt(2 * np.pi)
    if p == 1:
        if k == 0:
            return sigma * phi + mu * ndtr(alpha)
        if k == 1:
            return ndtr(alpha)
        he = np.polynomial.hermite_e.hermeval(alpha, [0] * (k - 2) + [1])
        return (-1) ** (k - 2) * sigma ** (1 - k) * he * phi
    if k >= p:
        import math

        return math.factorial(p) * wick(k - p + 1, 1, mu, sigma)
    if (k, p) == (1, 3):
        return 3 * ((mu**2 + sigma**2) * ndtr(alpha) + mu * sigma * phi)
    if (k, p) == (2, 3):
        return 6 * wick(0, 1, mu, sigma)
    raise ValueError((k, p))


def main():
    output = Path("results/phase2/ablation_v30_067_r02_full_k31_charged_component.json")
    assert not output.exists()
    ns, sources = extract_generators(Path("/tmp/whest-official-cumulant-reference"))
    classes = ns["get_all_terms_iso"](3, d_max=4, augment=True)[(3, 1)]
    rng = np.random.default_rng(6702)
    rows = []
    for n in (8, 32, 64):
        vectors = {v for part in classes for v in part}
        factors = {
            v: rng.normal(size=(n,) * sum(x > 0 for x in v)).astype("float32") for v in vectors
        }
        mu = rng.normal(size=n)
        sigma = rng.uniform(0.4, 2, size=n)
        pairs = {(sum(v[i] for v in part), p) for part in classes for i, p in enumerate((3, 1))}
        wicks = {pair: wick(*pair, mu, sigma).astype("float32") for pair in pairs}
        expected = np.zeros((n, n), dtype=np.float64)
        # Independent explicit per-index scalar calculation, same frozen diagram specification.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for part, count in classes.items():
                    value = float(coefficient(part, count))
                    for v in part:
                        index = tuple(x for x, support in zip((i, j), v) if support)
                        value *= float(factors[v][index])
                    value *= float(wicks[(sum(v[0] for v in part), 3)][i])
                    value *= float(wicks[(sum(v[1] for v in part), 1)][j])
                    expected[i, j] += value
        with flops.BudgetContext(flop_budget=2**35) as ctx:
            fs = {v: fnp.array(a) for v, a in factors.items()}
            ws = {pair: fnp.array(a) for pair, a in wicks.items()}
            actual = fnp.zeros((n, n), dtype=fnp.float32)
            for part, count in classes.items():
                value = fnp.ones((n, n), dtype=fnp.float32) * float(coefficient(part, count))
                for v in part:
                    shape = tuple(n if support else 1 for support in v)
                    value = value * fnp.reshape(fs[v], shape)
                value = value * ws[(sum(v[0] for v in part), 3)][:, None]
                value = value * ws[(sum(v[1] for v in part), 1)][None, :]
                actual = actual + value
            fnp.fill_diagonal(actual, 0)
        relative = float(np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected))
        assert relative < 2e-6
        rows.append(
            dict(
                n=n,
                terms=len(classes),
                relative_error=relative,
                charged_flops=ctx.summary_dict()["flops_used"],
            )
        )
    output.write_text(
        json.dumps(
            dict(
                rows=rows,
                sources=sources,
                mechanism="Complete augmented K31 diagram polynomial, distinct row/column output; all degree<=4 input slices supplied explicitly",
                scope="Legally charged exact dense component with offline synthetic input cumulants; input generation and Wick table preparation are oracle diagnostics, not complete candidate",
                limitations=[
                    "No full K211 production or centered conversion integration",
                    "No actual V29 state replacement or benchmark score",
                    "Current input wk431 in V29 remains proxy; component does not fix its provenance",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()
