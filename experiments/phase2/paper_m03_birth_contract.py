"""Independent NumPy dense contraction of official first-birth ordinary K3."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/tmp/whest-official-cumulant-reference/src')
import torch
from mlp_kprop.diagslice import diagslice
from mlp_kprop.kprop_harmonic import SIMPLE, mlp_kprop
from mlp_kprop.mlp import MLP


def main():
    path = Path('results/phase2/paper_m03_r14_birth_contract.json')
    assert not path.exists()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    previous = json.loads(Path('results/phase2/paper_m03_r13_state_v29.json').read_text())
    rows = []
    for seed in (13026031, 13026032):
        file = Path(f'benchmark/data/paper_m03_network64_{seed}.npz')
        weights = np.load(file)['weights'].astype(np.float32)
        mlp = MLP(input_dim=64, hidden_dim=64, output_dim=64, num_layers=5)
        for i in range(4):
            mlp.Ws[i].weight.copy_(torch.from_numpy(weights[i].T.copy()))
        out = mlp_kprop(mlp, {1: torch.zeros(64), 2: torch.eye(64)}, k_max=3,
                         kind=SIMPLE, factor=True, use_avg_metric=True,
                         output_all=True, up_to_layer='pre1')
        birth = out['act0'][3].to_tensor().numpy().astype(np.float64)
        W = weights[1].T.astype(np.float64)
        # U[i,b,c] = sum_a W[i,a] K[a,b,c]; V[i,j,c] = sum_b W[j,b] U[i,b,c].
        U = np.einsum('ia,abc->ibc', W, birth)
        V = np.einsum('jb,ibc->ijc', W, U)
        full = np.einsum('kc,ijc->ijk', W, V)
        diag = np.einsum('iii->i', full)
        pair = np.einsum('iij->ij', full).copy()
        np.fill_diagonal(pair, 0)
        official_diag = diagslice(out['pre1'][3], (3,)).numpy()
        official_pair = diagslice(out['pre1'][3], (2, 1), output_zero_repeated=True).numpy()
        local = next(r for r in previous['rows'] if r['seed'] == seed)
        local = next(r['objects'] for r in local['states'] if r['stage'] == 'pre' and r['layer'] == 1)
        artifact = Path(f'benchmark/data/paper_m03_r14_birth_objects_{seed}.npz')
        assert not artifact.exists()
        np.savez(artifact, ordinary_birth=birth, W=W, dense_diag=diag, dense_pair=pair,
                 official_diag=official_diag, official_pair=official_pair)
        rows.append(dict(seed=seed, input_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            dense_vs_official_diag_max=float(abs(diag-official_diag).max()),
            dense_vs_official_pair_max=float(abs(pair-official_pair).max()),
            dense_vs_v29_diag_mse=float(((diag-local['k3_diag'])**2).mean()),
            dense_vs_v29_pair_mse=float(((pair-local['k3_21'])**2).mean()),
            numpy_dense_analytic_multiply_add_flops=3*2*64**4,
            max_permutation_error=float(abs(birth-birth.transpose(1,0,2)).max())))
    path.write_text(json.dumps(dict(rows=rows, complete=True,
        official_commit='93d091a4c26c042bfffa28f2e76a81bc0aba94bb',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources=['factor_k3.py:FactoredTensor.to_tensor, contract_W, factored_nonlin_kprop_k3 step4',
                 'cumulants.py:DS_pK_to_K', 'diagslice.py:diagslice'],
        formula='Kprime_ijk=sum_abc Wia Wjb Wkc Kabc; D3=Kprime_iii; D21=Kprime_iij (zero i=j)',
        scope='Official returned birth is ordinary K after pK conversion. No K3 harmonic projection; K4 separate. Offline NumPy analytic cost is not flopscope competition billing. No truth or score.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
