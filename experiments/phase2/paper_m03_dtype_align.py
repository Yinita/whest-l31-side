"""K3 same quantized weights, dtype and paired reference; source mechanisms unproven."""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/tmp/whest-official-cumulant-reference/src')
import numpy as np
import torch
from mlp_kprop.flop_utils import NamedFlopCounter
from mlp_kprop.kprop_harmonic import AUGMENT, SIMPLE, mlp_kprop
from mlp_kprop.mlp import MLP


def main():
    path = Path('results/phase2/paper_m03_r08_dtype_align.json')
    assert not path.exists()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    v29 = json.loads(Path('results/phase2/paper_m03_r07_v29_same64.json').read_text())
    rows, references = [], []
    for seed in (13026031,13026032):
        file = Path(f'benchmark/data/paper_m03_network64_{seed}.npz')
        data = np.load(file)
        quantized = data['weights'].astype(np.float32)
        samples = 262144
        torch.set_default_dtype(torch.float64)
        mlp = MLP(input_dim=64,hidden_dim=64,output_dim=64,num_layers=5)
        for i in range(4):
            mlp.Ws[i].weight.copy_(torch.from_numpy(quantized[i].T.copy()).double())
        sums = torch.zeros(64)
        squares = torch.zeros(64)
        rng = torch.Generator().manual_seed(seed+2000)
        for _ in range(samples//1024):
            x = mlp(torch.randn(1024,64,generator=rng),up_to_layer='act3').out
            sums += x.sum(0)
            squares += (x*x).sum(0)
        mean = sums/samples
        variance = (squares-sums*mean)/(samples-1)
        baseline = np.array(next(r for r in v29['rows'] if r['seed']==seed)['prediction'])
        references.append(dict(seed=seed,samples=samples,reference_seed=seed+2000,
                               reference_sampling_mse=float(variance.mean()/samples),
                               v29_same_reference_noisy_mse=float(np.mean((baseline-mean.numpy())**2)),
                               quantized_weight_sha256=hashlib.sha256(quantized.tobytes()).hexdigest()))
        for dtype in (torch.float32,torch.float64):
            torch.set_default_dtype(dtype)
            mlp = mlp.to(dtype)
            for kind in (SIMPLE,AUGMENT):
                started = time.monotonic()
                try:
                    with NamedFlopCounter() as count:
                        state = mlp_kprop(mlp,{1:torch.zeros(64),2:torch.eye(64)},k_max=3,
                                          kind=kind,factor=True,use_avg_metric=True,
                                          up_to_layer='act3',output_d_max=1)
                    pred = state[1].to_tensor().double().numpy()
                    rows.append(dict(seed=seed,dtype=str(dtype),kind=kind.name,
                                     prediction=pred.tolist(),noisy_mse=float(np.mean((pred-mean.numpy())**2)),
                                     prediction_mse_vs_v29=float(np.mean((pred-baseline)**2)),
                                     paper_flops=count.total(),elapsed_s=time.monotonic()-started))
                except Exception as error:
                    rows.append(dict(seed=seed,dtype=str(dtype),kind=kind.name,error=repr(error)))
                path.write_text(json.dumps(dict(rows=rows,references=references,complete=False),indent=2)+'\n')
    path.write_text(json.dumps(dict(rows=rows,references=references,complete=True,
                                    scope='Same float32quantizedweights,reference float64forward,official K3factor float32/64. V29float32genericpath; paper vsflopscope计量different; no stagegain'),indent=2)+'\n')


if __name__ == '__main__':
    main()
