"""Small synthetic oracle checks for fourth-order harmonic core conventions.

Offline algebra only: full n**4 tensors must never enter submission code.
"""
import itertools
import json
from pathlib import Path

import numpy as np


def lift_core(g, metric):
    eye = np.eye(len(g)) * metric
    return sum(np.einsum(spec, g, eye) for spec in (
        'ij,kl->ijkl', 'ik,jl->ijkl', 'il,jk->ijkl',
        'jk,il->ijkl', 'jl,ik->ijkl', 'kl,ij->ijkl',
    )) / 6


def recover_core(tensor, metric):
    n = tensor.shape[0]
    trace = np.einsum('ijkk->ij', tensor)
    return (6 * trace / (metric * (n + 4))
            - 3 * np.trace(trace) * np.eye(n) / (metric * (n + 2) * (n + 4)))


def main():
    rng = np.random.default_rng(0)
    rows = []
    for n, metric in itertools.product((4, 8, 16), (1, 2)):
        raw = rng.normal(size=(n, n))
        g = (raw + raw.T) / 2
        tensor = lift_core(g, metric)
        recovered = recover_core(tensor, metric)
        trace = np.einsum('ijkk->ij', tensor)
        split_trace = np.zeros_like(trace)
        for i in range(n):
            for j in range(n):
                if i == j:
                    split_trace[i, j] = tensor[i, i, i, i] + sum(tensor[i, i, k, k] for k in range(n) if k != i)
                else:
                    split_trace[i, j] = tensor[i, i, i, j] + tensor[i, j, j, j] + sum(tensor[i, j, k, k] for k in range(n) if k not in (i, j))
        error = float(np.max(np.abs(recovered - g)))
        split_error = float(np.max(np.abs(split_trace - trace)))
        assert error < 1e-12 and split_error < 1e-12
        rows.append({'n': n, 'metric': metric, 'inverse_max_error': error, 'slice_trace_max_error': split_error})
    result = {
        'mode': 'synthetic algebra diagnostic; not benchmark accuracy or candidate',
        'seed': 0,
        'formula': 'H=trace(T); G=6H/[m(n+4)]-3 trace(H)I/[m(n+2)(n+4)]',
        'convention': 'Lift T=m*Sym(G tensor I), distinct from post-ReLU identity-metric projection convention',
        'source': 'https://arxiv.org/html/2605.05179v2#S4.SS2',
        'rows': rows,
        'implication': 'Off-diagonal H needs K31 and capped K211; diagonal K4/K22 alone cannot recover it. No claim about their competition error contribution.',
    }
    path = Path('results/phase2/harmonic_core_algebra.json')
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('Six inverse and slice checks passed; max error', max(x['inverse_max_error'] for x in rows))


if __name__ == '__main__':
    main()
