"""Independent bounded diagram enumeration and explicit production values."""

import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.phase2.audit_official_augmented_terms import extract_generators


def independent_parts(d, budget):
    vectors = []
    for v in itertools.product(range(5), repeat=d):
        s = sum(v)
        cost = s - 1 - int(all(x % 2 == 0 for x in v))
        if 2 <= s <= 4 and cost <= budget and not (s <= 2 and sum(x > 0 for x in v) <= 1):
            vectors.append((v, cost))
    vectors.sort()

    def walk(start, part, cost):
        if part:
            edges = [{i for i, x in enumerate(v) if x} for v in part]
            reached = {0}
            while True:
                expanded = reached | set().union(*(e for e in edges if e & reached))
                if expanded == reached:
                    break
                reached = expanded
            if len(reached) == d:
                yield tuple(part)
        for j in range(start, len(vectors)):
            v, c = vectors[j]
            if cost + c <= budget:
                yield from walk(j, part + [v], cost + c)

    return list(walk(0, [], 0))


def keeps(part, output):
    # K31/K211 outputs are below top slice: independent documented factor rule.
    return len(part) == 1 or (1, 1, 1) not in part


def main():
    root = Path(__file__).resolve().parents[2]
    ns, sources = extract_generators(Path("/tmp/whest-official-cumulant-reference"))
    rng = np.random.default_rng(63)
    rows = []
    for output in [(3, 1), (2, 1, 1)]:
        d = len(output)
        for augment in [False, True]:
            official_raw = [
                vp for ip, vp in ns["get_all_terms"](3, d_max=4, augment=augment) if ip == output
            ]
            independent = independent_parts(d, 2 + int(augment))
            assert set(official_raw) == set(independent)
            official_iso = ns["get_all_terms_iso"](3, d_max=4, augment=augment)[output]
            for n in [4, 6, 8]:
                # Arbitrary nonzero cumulant slice tensors; shared by independent evaluation routes.
                factors = {
                    v: rng.normal(size=(n,) * sum(x > 0 for x in v))
                    for vp in independent
                    for v in vp
                }
                wick = {k: rng.normal(size=(n,)) for k in range(7)}
                for filtered in [False, True]:
                    expected = np.zeros((n,) * d)
                    for vp in independent:
                        if filtered and not keeps(vp, output):
                            continue
                        derivative = tuple(sum(v[i] for v in vp) for i in range(d))
                        coefficient = 1 / math.prod(
                            math.factorial(c) * math.prod(math.factorial(x) for x in v) ** c
                            for v, c in Counter(vp).items()
                        )
                        for indices in itertools.product(range(n), repeat=d):
                            if len(set(indices)) != d:
                                continue
                            value = coefficient
                            for v in vp:
                                value *= factors[v][tuple(indices[i] for i, x in enumerate(v) if x)]
                            for axis in range(d):
                                value *= wick[derivative[axis]][indices[axis]]
                            expected[indices] += value
                    actual = np.zeros_like(expected)
                    # Official canonical classes: expand all distinct output-power-preserving axis permutations.
                    for canonical, count in official_iso.items():
                        if filtered and not ns["factored_keeps_term"](3, output, canonical):
                            continue
                        orbit = set()
                        for perm in itertools.permutations(range(d)):
                            if tuple(output[i] for i in perm) != output:
                                continue
                            orbit.add(tuple(sorted(tuple(v[i] for i in perm) for v in canonical)))
                        assert len(orbit) == count
                        for vp in orbit:
                            coef = 1 / math.prod(
                                math.factorial(c) * math.prod(math.factorial(x) for x in v) ** c
                                for v, c in Counter(vp).items()
                            )
                            value = np.ones((n,) * d) * coef
                            for v in vp:
                                support = [i for i, x in enumerate(v) if x]
                                shape = [n if i in support else 1 for i in range(d)]
                                value *= factors[v].reshape(shape)
                            for axis in range(d):
                                k = sum(v[axis] for v in vp)
                                shape = [1] * d
                                shape[axis] = n
                                value *= wick[k].reshape(shape)
                            actual += value
                    mask = np.zeros_like(actual, dtype=bool)
                    for indices in itertools.product(range(n), repeat=d):
                        mask[indices] = len(set(indices)) == d
                    actual *= mask
                    err = float(np.max(abs(actual - expected)))
                    assert err < 1e-10
                    caperr = None
                    if d == 3:
                        caperr = float(np.max(abs(actual.sum(axis=0) - expected.sum(axis=0))))
                    rows.append(
                        dict(
                            output=output,
                            augment=augment,
                            filtered=filtered,
                            n=n,
                            raw_terms=sum(not filtered or keeps(vp, output) for vp in independent),
                            iso_terms=sum(
                                not filtered or ns["factored_keeps_term"](3, output, vp)
                                for vp in official_iso
                            ),
                            max_error=err,
                            capped_max_error=caperr,
                        )
                    )
    path = root / "results/phase2/ablation_v30_063_064_production_parity.json"
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                scope="Independent bounded vector multiset enumeration vs official canonical generator; arbitrary cumulant/Wick arrays, strict distinct indices, rawfull and factoredfilter separately",
                sources=sources,
                rows=rows,
                limitations=[
                    "Official Torch factor_k3/factor_k4 kernel execution not covered",
                    "Nonlinear power production only; ordinary cumulant conversion checked separately",
                    "No benchmark score",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    print("rows", len(rows), "maxerror", max(r["max_error"] for r in rows))


if __name__ == "__main__":
    main()
