"""Same frozen64x4 inputs, V29 off-suite base path; charged local diagnostic."""
import hashlib
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext
from whestbench.domain import MLP


def main():
    path = Path('results/phase2/paper_m03_r18_exact21_capture.json')
    assert not path.exists()
    source = Path('estimators/k3_v29/estimator.py')
    rows = []
    for seed in (13026031,13026032):
        file = Path(f'benchmark/data/paper_m03_network64_{seed}.npz')
        reference = np.load(file)
        spec = importlib.util.spec_from_file_location('v29small',source)
        module = importlib.util.module_from_spec(spec)
        captured = []
        def capture_birth(layer, **objects):
            if layer == 0:
                captured.append({key: value.tolist() for key, value in objects.items()})
        module.capture_birth = capture_birth
        code = source.read_text().replace("            newborn = (a_b, Rr_full, Lr_full, S3c, e_b, Ff_b)", "            capture_birth(li, A=a_b, R=Rr_full, L=lb[k_b], s=S3c, e=e_b, w2=w2, residual=Rres, marginal_k3=K3v, marginal_k21=K21)\n            newborn = (a_b, Rr_full, Lr_full, S3c, e_b, Ff_b)")
        exec(compile(code, str(source), 'exec'), module.__dict__)
        estimator = module.Estimator()
        with flops.BudgetContext(flop_budget=2**41):
            estimator.setup(SetupContext(width=64,depth=4,flop_budget=2**41,api_version='0.1',seed=0))
            weights = fnp.asarray(reference['weights'].astype(np.float32))
            mlp = MLP(width=64,depth=4,weights=[weights[i] for i in range(4)],seed=seed)
        with flops.BudgetContext(flop_budget=2**41) as ctx:
            pred = estimator.predict(mlp,2**41)
        with flops.BudgetContext(flop_budget=2**41):
            values = np.array(pred.tolist())
        rows.append(dict(seed=seed,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         input_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                         birth=captured[0], prediction=values[-1].tolist(),noisy_mse=float(np.mean((values[-1]-reference['reference_mean'])**2)),
                         budget_summary=ctx.summary_dict()))
    path.write_text(json.dumps(dict(rows=rows,scope='V29 off-suite shape-generic path; float32 input rounding vs float64 official paper; charged local not competition score'),indent=2)+'\n')


if __name__ == '__main__':
    main()
