"""Freeze all eight affine artifacts and compare charged float32 inference."""
import ast
import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
    report = Path('results/phase2/ablation_v30_128_r02_allfold_contract.json')
    assert not report.exists()
    order = ['ones', 'mu', 'var', 'sigma', 'alpha', 'abs_alpha', 'phi', 'Phi', 'prediction', 'D3', 'D21norm', 'K4', 'sigma_phi']
    rows = []
    for held in range(8):
        folder = Path(f'estimators/ablation_v30_098_fold{held}')
        source = folder / 'estimator.py'
        tree = ast.parse(source.read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ['FOLD0_COEFFICIENT', 'FOLD0_INTERCEPT']:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
        x = np.load(f'benchmark/data/v29_features_v30_096/network_{held}.npz')['features'][-1].astype('float32')
        coef = np.array(constants['FOLD0_COEFFICIENT'], dtype='float32')
        intercept = np.float32(constants['FOLD0_INTERCEPT'])
        with flops.BudgetContext(flop_budget=2**30) as ctx:
            actual = fnp.array(x) @ fnp.array(coef) + float(intercept)
        expected = x.astype('float64') @ coef.astype('float64') + float(intercept)
        err = float(np.max(abs(np.asarray(actual)-expected)))
        assert err < 1e-7
        provenance = json.loads((folder/'provenance.json').read_text())
        assert held not in provenance['train_ids']
        rows.append(dict(held=held, feature_order=order, normalization='mean/scale folded into affinecoef/intercept; rawfeatures at runtime', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), constants_sha256=hashlib.sha256(json.dumps(constants,sort_keys=True).encode()).hexdigest(), maximum_absolute_error=err, flops=ctx.summary_dict()['flops_used'], train_ids=provenance['train_ids']))
    report.write_text(json.dumps(dict(rows=rows, scope='Allfrozen foldartifact inference contract; training IDs only provenance, no runtimeID/target table. NegativeCV does not prevent correctness checks; no model quality claim'),indent=2)+'\n')
    print(max(x['maximum_absolute_error'] for x in rows))


if __name__ == '__main__':
    main()
