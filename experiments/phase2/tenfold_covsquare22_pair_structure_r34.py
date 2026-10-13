"""Exact Gaussian C-square repeated22 matching by matrix pair, scope audited."""
# ruff: noqa: E402
import os

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
import hashlib
import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def sympair(a, b):
    return sum(np.einsum('ij,kl->ijkl', a, b).transpose(p)
               for p in itertools.permutations(range(4)))/24


def d22_pair(a, b):
    return (fnp.diag(a)[:, None]*fnp.diag(b)[None, :]
            +fnp.diag(b)[:, None]*fnp.diag(a)[None, :]+4*a*b)/6


def exclusive(d):
    n = len(d)
    result = np.zeros((n,)*4)
    for i, j in itertools.combinations(range(n), 2):
        for p in set(itertools.permutations((i, i, j, j))):
            result[p] = d[i, j]
    return result


def main():
    output = Path('results/phase2/tenfold_covsquare22_pair_structure_r34.json')
    assert not output.exists()
    rng = np.random.default_rng(34001)
    rows = []
    for n in (4, 8, 12):
        c = rng.normal(0, .08, (n, n));c = (c+c.T)/2;np.fill_diagonal(c, 0)
        f = rng.normal(0, 1, n)
        d = f[:, None]*f[None, :]*c*c
        a, b = f[:, None]*c*f[None, :], 1.5*c
        dense = sympair(a, b)
        only = exclusive(d)
        w = rng.normal(0, 1, (n, n))
        moved = np.einsum('ia,jb,kc,ld,abcd->ijkl',w,w,w,w,dense,optimize=True)
        movedonly = np.einsum('ia,jb,kc,ld,abcd->ijkl',w,w,w,w,only,optimize=True)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            fa, fb, fw = [fnp.asarray(v) for v in (a, b, w)]
            born22 = d22_pair(fa, fb)
            moved22 = d22_pair(fw@fa@fw.T, fw@fb@fw.T)
        err = max(float(np.max(abs(np.asarray(born22)-d))),
                  float(np.max(abs(np.asarray(moved22)-np.einsum('iijj->ij',moved)))))
        assert err < 1e-10
        # Paired carry is exact for its represented tensor, but differs from
        # the strict 22-only lift on 211/1111 and after transport.
        rows.append({'n':n,'born22maxerror':float(np.max(abs(np.asarray(born22)-d))),
                     'pair_transport22maxerror':float(np.max(abs(np.asarray(moved22)-np.einsum('iijj->ij',moved)))),
                     'pair_vs_exclusive_tensor_relative_norm':float(np.linalg.norm(dense-only)/np.linalg.norm(only)),
                     'pair_vs_exclusive_transported22_relative_norm':float(np.linalg.norm(np.asarray(moved22)-np.einsum('iijj->ij',movedonly))/np.linalg.norm(np.einsum('iijj->ij',movedonly))),
                     'legal_pair_component_flops':ctx.summary_dict()['flops_used']})
    n, depth, pairs = 1024, 16, 2
    # each symmetric matrix sandwich costs <=4n3. two matrices per pair.
    plain = 8*n**3*pairs*(depth*(depth-1)//2)
    report={'complete':True,'rows':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'structure':'d22_ij=f_i*f_j*Cij² (Cdiag0): A=diag(f)Cdiag(f), B=1.5C, tensorSym(A⊗B) matches all repeated22 exactly at birth, repeated31/4 zero.',
            'two_gaussian_Csquare_channels_plain_transport_flops':plain,
            'two_channels_symmetric_output_transport_estimate_flops':int(plain*.875),
            'scope':'Target-free structure contract only. Matrixpair automatically adds nonzero211/1111; not exact22-exclusive transport or fullGaussianK4 producer. Plainallhistorytwochannels transport≈2.06T beforeparent/feed exceedsbudget, but legalStrassen lowerbill may differ. No score/candidate claim.'}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
