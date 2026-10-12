"""Official full variants on one frozen small network; offline method diagnostic."""
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/tmp/whest-official-cumulant-reference/src')
import torch
from mlp_kprop.kprop_harmonic import AUGMENT, SIMPLE, mlp_kprop
from mlp_kprop.mlp import MLP


def main():
    path = Path('results/phase2/paper_m03_r03_small_variants.json')
    assert not path.exists()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(13026001)
    n = 8
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=3)
    source_hash = hashlib.sha256(b''.join(w.weight.detach().numpy().tobytes() for w in mlp.Ws)).hexdigest()
    generator = torch.Generator().manual_seed(13026002)
    sums = torch.zeros(n)
    squares = torch.zeros(n)
    samples = 262144
    with torch.no_grad():
        for _ in range(samples//1024):
            value = mlp(torch.randn(1024, n, generator=generator)).out
            sums += value.sum(0)
            squares += (value*value).sum(0)
    mean = sums/samples
    variance = (squares-sums*mean)/(samples-1)
    rows = []
    for order in (2, 3, 4):
        for kind in (SIMPLE, AUGMENT):
            for factor in (False, True):
                started = time.monotonic()
                try:
                    out = mlp_kprop(mlp, {1:torch.zeros(n), 2:torch.eye(n)},
                                    k_max=order, kind=kind, factor=factor,
                                    use_avg_metric=True, output_d_max=1)
                    prediction = out[1].to_tensor().detach()
                    row = dict(order=order, kind=kind.name, factor=factor,
                               prediction=prediction.tolist(),
                               noisy_mse=float(((prediction-mean)**2).mean()))
                except Exception as error:
                    row = dict(order=order, kind=kind.name, factor=factor, error=repr(error))
                row['elapsed_s'] = time.monotonic()-started
                rows.append(row)
                path.write_text(json.dumps(dict(rows=rows, complete=False),indent=2)+'\n')
                print(row,flush=True)
    path.write_text(json.dumps(dict(rows=rows, complete=True, width=n, layers=3,
                                    output='final linear output; 2 ReLU hidden layers',
                                    network_sha256=source_hash, seed=13026001,
                                    reference_seed=13026002, samples=samples,
                                    reference_sampling_mse=float(variance.mean()/samples),
                                    official_commit='93d091a4c26c042bfffa28f2e76a81bc0aba94bb',
                                    versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy']},
                                    scope='Offline official Torch implementations; not flopscope candidate or competition adjusted score; weak MC reference'),indent=2)+'\n')


if __name__ == '__main__':
    main()
