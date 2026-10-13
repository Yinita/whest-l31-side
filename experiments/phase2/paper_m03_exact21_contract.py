"""Single offline replacement of rank16 Rres approximation by exact captured Rres."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sym(tensor):
    return sum(tensor.transpose(p) for p in itertools.permutations(range(3)))/6


def main():
    path = Path('results/phase2/paper_m03_r18_exact21_contract.json')
    assert not path.exists()
    capture_path = Path('results/phase2/paper_m03_r18_exact21_capture.json')
    capture = json.loads(capture_path.read_text())
    rows = []
    for row in capture['rows']:
        seed = row['seed']
        b = {k:np.asarray(v,dtype=np.float64) for k,v in row['birth'].items()}
        reference_path = Path(f'benchmark/data/paper_m03_r14_birth_objects_{seed}.npz')
        official = np.load(reference_path)
        identity = np.eye(64)
        hub = sym(np.einsum('ir,jr,kr->ijk',3*b['A'],b['A']*b['w2'][None,:],identity))
        variants = {}
        for label,residual in [('rank16',b['R']@b['L'].T),('exact',3*b['residual'].T)]:
            matrix = np.diag(b['s'])+3*b['A']*b['e'][None,:]+residual
            birth = hub+sym(np.einsum('ir,jr,kr->ijk',matrix,identity,identity))
            diff = birth-official['ordinary_birth']
            diagonal = np.einsum('iii->i',birth)
            pair = np.einsum('iij->ij',birth).copy()
            np.fill_diagonal(pair,0)
            W = official['W']
            U = np.einsum('ia,abc->ibc',W,birth)
            V = np.einsum('jb,ibc->ijc',W,U)
            transported = np.einsum('kc,ijc->ijk',W,V)
            variants[label] = dict(birth_mse=float((diff**2).mean()),birth_max=float(abs(diff).max()),
                birth_diag_vs_marginal_max=float(abs(diagonal-b['marginal_k3']).max()),
                birth_pair_vs_marginal_max=float(abs(pair-b['marginal_k21']).max()),
                pre1_diag_vs_official_mse=float(((np.einsum('iii->i',transported)-official['official_diag'])**2).mean()))
        rows.append(dict(seed=seed,variants=variants,input_sha256=row['input_sha256'],
            official_objects_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            charged_capture_flops=row['budget_summary']['flops_used']))
    path.write_text(json.dumps(dict(rows=rows,complete=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        intervention='M residual RLtranspose replaced with3Rres.transpose only; all other birth objects frozen',
        formula='M_exact=diag(s)+3A diag(e)+3Rres.transpose; T=Sym(3A,A diag(w2),I)+Sym(M,I,I)',
        sources=['V29 source1414-1436 S21/S_sep/Rres and M comments','r14 official ordinary K3 objects'],
        scope='Offline diagnostic, no oracle target; no candidate score or deployment cost. Two dense transports analytic201326592 FLOPs/network distinct from flopscope.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
