"""Independent partition enumeration verifies complete power-to-K211 conversion."""
import json
import math
from pathlib import Path

import numpy as np


def partitions(items):
    if not items:
        yield []
        return
    first, *rest = items
    for partition in partitions(rest):
        yield [(first,)] + partition
        for index in range(len(partition)):
            yield partition[:index] + [(first,) + partition[index]] + partition[index+1:]


def cumulant(vectors, weights):
    total = 0.0
    for partition in partitions(list(range(len(vectors)))):
        product = 1.0
        for block in partition:
            product *= np.dot(weights, np.prod([vectors[i] for i in block], axis=0))
        total += (-1)**(len(partition)-1) * math.factorial(len(partition)-1) * product
    return total


def main():
    rng = np.random.default_rng(42)
    rows = []
    for n in (4, 7):
        values = rng.normal(size=(n, 53)) + rng.uniform(-2, 2, size=(n, 1))
        weights = rng.uniform(size=53)
        weights /= weights.sum()
        means = values @ weights
        centered = values - means[:, None]
        cov = (centered * weights) @ centered.T
        direct = np.zeros((n, n))
        converted = np.zeros_like(direct)
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                for i in range(n):
                    if i in (j, k):
                        continue
                    x, y, z = values[[i, j, k]]
                    pk211 = cumulant([x*x, y, z], weights)
                    pk111 = cumulant([x, y, z], weights)
                    k211 = cumulant([x, x, y, z], weights)
                    formula = pk211 - 2*means[i]*pk111 - 2*cov[i, j]*cov[i, k]
                    direct[j, k] += k211
                    converted[j, k] += formula
        error = float(np.max(np.abs(direct-converted)))
        assert error < 2e-12
        rows.append({'n': n, 'support_points': 53, 'capped_conversion_max_error': error})
    out = Path('results/phase2/k211_conversion_algebra.json')
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps({'mode': 'exact finite-distribution algebra diagnostic, no MC reference',
        'formula': 'K_iijk = cum(X_i^2,X_j,X_k) - 2 E[X_i] cum(X_i,X_j,X_k) - 2 Cov(X_i,X_j) Cov(X_i,X_k)',
        'capped_formula': 'sum over i distinct from j,k; diagonal j=k excluded',
        'method': 'Independent cumulant set-partition enumeration versus power-cumulant conversion on arbitrary shifted non-Gaussian finite distributions',
        'source': 'https://arxiv.org/html/2605.05179v2#S3.SS3',
        'rows': rows, 'limitations': ['Conversion exact, but approximate production of power cumulants not validated', 'Does not establish end-to-end score improvement']}, indent=2)+'\n')
    print('Two complete capped conversion checks passed', rows)


if __name__ == '__main__':
    main()
