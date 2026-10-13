"""Independent dense reconstruction of frozen V29 first birth and slices."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sym(t):
    return sum(t.transpose(p) for p in itertools.permutations(range(3)))/6


def main():
    path = Path('results/phase2/paper_m03_r17_repeated_transport.json')
    assert not path.exists()
    capture = json.loads(Path('results/phase2/paper_m03_r15_birth_capture.json').read_text())
    previous = json.loads(Path('results/phase2/paper_m03_r13_state_v29.json').read_text())
    rows = []
    for row in capture['rows']:
        seed = row['seed']
        b = {k: np.array(v, dtype=np.float64) for k, v in row['birth'].items()}
        identity = np.eye(64)
        M = np.diag(b['s'])+3*b['A']*b['e'][None,:]+b['R']@b['L'].T
        birth = sym(np.einsum('ir,jr,kr->ijk',3*b['A'],b['A']*b['w2'][None,:],identity))
        birth += sym(np.einsum('ir,jr,kr->ijk',M,identity,identity))
        official = np.load(f'benchmark/data/paper_m03_r14_birth_objects_{seed}.npz')
        W = official['W']
        difference = birth-official['ordinary_birth']
        i,j,k = np.indices((64,64,64))
        masks = dict(all_distinct=(i!=j)&(j!=k)&(i!=k), all_equal=(i==j)&(j==k))
        masks['exactly_two_equal'] = ~(masks['all_distinct']|masks['all_equal'])
        contributions = {}
        transported_parts = {}
        for key, mask in masks.items():
            residual = difference*mask
            U = np.einsum('ia,abc->ibc',W,residual)
            V = np.einsum('jb,ibc->ijc',W,U)
            T = np.einsum('kc,ijc->ijk',W,V)
            diag = np.einsum('iii->i',T)
            pair = np.einsum('iij->ij',T).copy()
            np.fill_diagonal(pair,0)
            transported_parts[key] = (diag,pair)
            contributions[key] = dict(diag_mse=float((diag**2).mean()),
                pair_mse=float((pair**2).mean()),diag_max=float(abs(diag).max()),pair_max=float(abs(pair).max()))
        d,p = transported_parts['all_equal']
        e,q = transported_parts['exactly_two_equal']
        combined_diag = d+e
        combined_pair = p+q
        rows.append(dict(seed=seed,input_sha256=row['input_sha256'],
            contributions=contributions,
            diagonal_pair_cross_diag=float(2*np.mean(d*e)),
            diagonal_pair_cross_pair=float(2*np.mean(p*q)),
            combined_diag_mse=float((combined_diag**2).mean()),
            combined_pair_mse=float((combined_pair**2).mean()),
            analytic_dense_transport_flops=3*100663296,
            baseline_charged_flops=row['budget_summary']['flops_used']))
    path.write_text(json.dumps(dict(rows=rows,complete=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        formula='Delta=V29birth-officialbirth; split iii/iij/distinct; transport each with W tensor W tensor W; squared error includes cross terms',
        sources=['V29 _dslices D3/D21 formulas','V29 newborn/Rr_full/lb assembly'],
        scope='Offline float64 dense algebra; analytic transport100663296 FLOPs not competition billing. No truth or score.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
