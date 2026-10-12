"""Compare final scratch reuse against frozenV29 with poisoned dead buffers."""
# ruff: noqa: E402
import os

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, SetupContext


def main():
    report = Path('results/phase2/ablation_v30_028_r02_poison_contract.json')
    assert not report.exists()
    rng = np.random.default_rng(2802)
    weights = [rng.normal(0, np.sqrt(2/32), (32, 32)).astype('float32') for _ in range(4)]
    rows = []
    for source in ('estimators/k3_v29/estimator.py', 'estimators/ablation_v30_028_r02/estimator.py'):
        spec = importlib.util.spec_from_file_location('c28', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        est = module.Estimator()
        with flops.BudgetContext(flop_budget=2**41):
            est.setup(SetupContext(width=32, depth=4, flop_budget=2**41, api_version='0.1', seed=0))
        original = est._dslices
        def poisoned(*args, **kwargs):
            if not kwargs.get('need_d21', True):
                bufs = args[11]
                fnp.copyto(bufs['lap'], fnp.full(bufs['lap'].shape, 12345.0, dtype=fnp.float32))
            return original(*args, **kwargs)
        est._dslices = poisoned
        with flops.BudgetContext(flop_budget=2**41) as ctx:
            values = est.predict(MLP(width=32, depth=4, weights=[fnp.array(w) for w in weights]), 2**41)
        rows.append(dict(source=source, values=np.asarray(values).tolist(), flops=ctx.summary_dict()['flops_used']))
    assert np.array_equal(rows[0]['values'], rows[1]['values'], equal_nan=True)
    assert rows[0]['flops'] == rows[1]['flops']
    report.write_text(json.dumps(dict(rows=rows, scope='Smallshape finaldeadLA/LP poisoned equally; equalNaN retained, not accuracy acceptance'), indent=2)+'\n')
    print('poison equality', rows[0]['flops'])


if __name__ == '__main__':
    main()
