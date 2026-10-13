"""Independent dense reconstruction of frozen V29 first birth and slices."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sym(t):
    return sum(t.transpose(p) for p in itertools.permutations(range(3)))/6


def main():
    path = Path('results/phase2/paper_m03_r15_birth_dense.json')
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
        U = np.einsum('ia,abc->ibc',W,birth)
        V = np.einsum('jb,ibc->ijc',W,U)
        T = np.einsum('kc,ijc->ijk',W,V)
        diag = np.einsum('iii->i',T)
        pair = np.einsum('iij->ij',T).copy()
        np.fill_diagonal(pair,0)
        local = next(r for r in previous['rows'] if r['seed']==seed)
        local = next(r['objects'] for r in local['states'] if r['stage']=='pre' and r['layer']==1)
        rows.append(dict(seed=seed,input_sha256=row['input_sha256'],source_sha256=row['source_sha256'],
            dense_vs_v29_diag_max=float(abs(diag-local['k3_diag']).max()),
            dense_vs_v29_pair_max=float(abs(pair-local['k3_21']).max()),
            represented_vs_official_birth_mse=float(((birth-official['ordinary_birth'])**2).mean()),
            represented_vs_official_birth_max=float(abs(birth-official['ordinary_birth']).max()),
            charged_capture_flops=row['budget_summary']['flops_used']))
    path.write_text(json.dumps(dict(rows=rows,complete=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        formula='T=Sym(3A,A diag(w2),I)+Sym(M,I,I); M=diag(s)+3A diag(e)+RLtranspose',
        sources=['V29 _dslices D3/D21 formulas','V29 newborn/Rr_full/lb assembly'],
        scope='Offline float64 dense algebra; analytic transport100663296 FLOPs not competition billing. No truth or score.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
