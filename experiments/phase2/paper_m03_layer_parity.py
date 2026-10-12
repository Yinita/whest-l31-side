"""Offline layer means: frozen official K3 versus unchanged V29, no score claim."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('backend', choices=['v29', 'official'])
    args = parser.parse_args()
    path = Path(f'results/phase2/paper_m03_r11_layer_{args.backend}.json')
    assert not path.exists()
    rows = []
    for seed in (13026031, 13026032):
        file = Path(f'benchmark/data/paper_m03_network64_{seed}.npz')
        weights = np.load(file)['weights'].astype(np.float32)
        identity = dict(seed=seed, input_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                        quantized_weight_sha256=hashlib.sha256(weights.tobytes()).hexdigest())
        if args.backend == 'v29':
            import flopscope as flops
            import flopscope.numpy as fnp
            from whestbench import SetupContext
            from whestbench.domain import MLP
            source = Path('estimators/k3_v29/estimator.py')
            spec = importlib.util.spec_from_file_location('baseline', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            estimator = module.Estimator()
            with flops.BudgetContext(flop_budget=2**41):
                estimator.setup(SetupContext(width=64, depth=4, flop_budget=2**41, api_version='0.1', seed=0))
                w = fnp.asarray(weights)
                mlp = MLP(width=64, depth=4, weights=[w[i] for i in range(4)], seed=seed)
            with flops.BudgetContext(flop_budget=2**41) as ctx:
                prediction = estimator.predict(mlp, 2**41)
            with flops.BudgetContext(flop_budget=2**41):
                means = prediction.tolist()
            rows.append(dict(**identity, kind='V29', layer_means=means,
                             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             budget_summary=ctx.summary_dict()))
        else:
            sys.path.insert(0, '/tmp/whest-official-cumulant-reference/src')
            import torch
            from mlp_kprop.kprop_harmonic import AUGMENT, SIMPLE, mlp_kprop
            from mlp_kprop.mlp import MLP
            torch.set_grad_enabled(False)
            torch.set_num_threads(1)
            torch.set_default_dtype(torch.float32)
            mlp = MLP(input_dim=64, hidden_dim=64, output_dim=64, num_layers=5)
            for i in range(4):
                mlp.Ws[i].weight.copy_(torch.from_numpy(weights[i].T.copy()))
            for kind in (SIMPLE, AUGMENT):
                output = mlp_kprop(mlp, {1: torch.zeros(64), 2: torch.eye(64)},
                                   k_max=3, kind=kind, factor=True, use_avg_metric=True,
                                   up_to_layer='act3', output_all=True)
                rows.append(dict(**identity, kind=kind.name,
                                 layer_means=[output[f'act{i}'][1].to_tensor().tolist() for i in range(4)],
                                 pre_means=[output[f'pre{i}'][1].to_tensor().tolist() for i in range(4)]))
        path.write_text(json.dumps(dict(rows=rows, complete=False), indent=2)+'\n')
    path.write_text(json.dumps(dict(rows=rows, complete=True,
        scope='No reference or oracle. Earliest mean divergence only; covariance/source attribution pending; off-suite V29 generic path.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
