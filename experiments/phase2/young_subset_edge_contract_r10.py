"""Independent dense symmetric CP oracle for the actual candidate helper."""
import importlib.util
import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.audit_official_augmented_terms import coefficient, extract_generators


def main():
    output = Path('results/phase2/ablation_v30_059_r10_subset_contract.json')
    assert not output.exists()
    spec = importlib.util.spec_from_file_location('subset', 'estimators/ablation_v30_059_subset_r10/estimator.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ns, sources = extract_generators(Path('/tmp/whest-official-cumulant-reference'))
    part = ((0, 1, 1), (1, 1, 1))
    count = ns['get_all_terms_iso'](3, d_max=4, augment=True)[(1, 1, 1)][part]
    assert coefficient(part, count) == 3
    rng = np.random.default_rng(5908)
    rows = []
    for n in (4, 8, 16):
        a, p, w = rng.normal(size=(3, n, n)).astype('float32')
        birth, w1, w2 = rng.normal(size=(3, n)).astype('float32')
        cov = rng.normal(size=(n, n)).astype('float32')
        cov = (cov + cov.T) / 2
        np.fill_diagonal(cov, 0)
        legs = [3 * a[:, :1] * w1[:, None], p[:, :1] * w1[:, None], a[:, :1] * birth[None, :1] * w1[:, None]]
        tensor = sum(np.einsum('ir,jr,kr->ijk', *perm, dtype=np.float64)
                     for perm in itertools.permutations(legs)) / 6
        edge = tensor * (w1[:, None, None] * w2[None, :, None] * w2[None, None, :] * cov[None, :, :]
                         + w2[:, None, None] * w1[None, :, None] * w2[None, None, :] * cov[:, None, :]
                         + w2[:, None, None] * w2[None, :, None] * w1[None, None, :] * cov[:, :, None])
        indices = np.arange(n)
        mask = ((indices[:, None, None] != indices[None, :, None])
                & (indices[:, None, None] != indices[None, None, :])
                & (indices[None, :, None] != indices[None, None, :]))
        expected = np.einsum('oi,oj,ok,ijk->o', w, w, w, edge * mask, dtype=np.float64)
        with flops.BudgetContext(flop_budget=2**41) as ctx:
            actual = module._young_subset_edge(*[fnp.array(x) for x in (w, a, p, birth, w1, w2, cov)])
        error = np.max(abs(np.asarray(actual) - expected)) / max(1, np.max(abs(expected)))
        assert error < 1e-5, error
        rows.append(dict(n=n, relative_error=float(error), flops=ctx.summary_dict()['flops_used']))
    output.write_text(json.dumps(dict(rows=rows, official_orbit_count=count,
        official_coefficient=str(coefficient(part, count)), sources=sources,
        scope='Synthetic symmetric B1 subset strict-distinct contract; not full source coverage or benchmark'), indent=2) + '\n')
    print(rows)


if __name__ == '__main__':
    main()
