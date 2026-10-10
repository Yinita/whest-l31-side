"""Verify factored K3 weighted contraction; offline dense tensor is diagnostic only."""
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(0)
    rows = []
    for n, rank in ((4, 3), (8, 5), (12, 7)):
        a, b, c = rng.normal(size=(3, n, rank))
        u, scale = rng.normal(size=(2, n))
        tensor = sum(np.einsum('it,jt,kt->ijk', x, y, z) for x, y, z in itertools.permutations((a, b, c))) / 6
        direct = np.einsum('i,ijk->jk', u, tensor) * scale[:, None] * scale[None, :]
        rotated = sum((y * (x.T @ u)[None, :]) @ z.T for x, y, z in ((a, b, c), (b, c, a), (c, a, b)))
        fast = (rotated + rotated.T) / 6 * scale[:, None] * scale[None, :]
        error = float(np.max(np.abs(direct-fast)))
        assert error < 1e-12
        # Repeated-index exclusions cannot be silently omitted from a capped sum.
        capped = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                if j != k:
                    capped[j, k] = sum(u[i]*tensor[i, j, k] for i in range(n) if i not in (j, k))*scale[j]*scale[k]
        corrected = fast.copy()
        for j in range(n):
            for k in range(n):
                corrected[j, k] -= (u[j]*tensor[j, j, k]+u[k]*tensor[k, j, k])*scale[j]*scale[k]
        np.fill_diagonal(corrected, 0)
        cap_error = float(np.max(np.abs(capped-corrected)))
        assert cap_error < 1e-12
        rows.append({'n': n, 'rank': rank, 'full_contraction_error': error, 'capped_contraction_error': cap_error})
    out = Path('results/phase2/source_contraction_algebra.json')
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps({'mode': 'synthetic algebra diagnostic, not candidate accuracy', 'seed': 0, 'rows': rows,
        'source': 'upstream lean/lean_k3_aug.py k211_from_sources_naive; commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d',
        'limitations': ['Validates weighted K3 tensor contraction only, not full nonlinear K211 production', 'Source inclusion/exclusion and conversion coefficients still need derivation', 'Dense n**3 tensors used only offline']}, indent=2)+'\n')
    print('Three full/capped contraction checks passed')


if __name__ == '__main__':
    main()
