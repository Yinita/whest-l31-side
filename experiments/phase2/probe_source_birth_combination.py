"""Verify old-source Wick, identity-born source and repeated-index corrections."""
import itertools
import json
from pathlib import Path

import numpy as np


def tensor(a, b, c):
    return sum(np.einsum('it,jt,kt->ijk', x, y, z) for x, y, z in itertools.permutations((a, b, c))) / 6


def contract(a, b, c, u, scale):
    acc = sum((y*(x.T@u)[None, :])@z.T for x, y, z in ((a,b,c),(b,c,a),(c,a,b)))
    return (acc+acc.T)/6*scale[:, None]*scale[None, :]


def repeated(t):
    n = len(t)
    d3 = np.array([t[i,i,i] for i in range(n)])
    d21 = np.array([[t[i,i,j] if i != j else 0 for j in range(n)] for i in range(n)])
    return np.diag(d3)+3*d21.T


def cap(t, u):
    n = len(t)
    return np.array([[sum(u[i]*t[i,j,k] for i in range(n) if i not in (j,k)) if j != k else 0 for k in range(n)] for j in range(n)])


def main():
    rng = np.random.default_rng(7)
    rows = []
    for n in (4,7,11):
        a,b,c = rng.normal(size=(3,n,3))
        x,y = rng.normal(size=(2,n,n))
        eye = np.eye(n)
        old = tensor(a,b,c)
        born = tensor(x,eye,y)
        w, ua, mean = rng.normal(size=(3,n))
        post_old = old*w[:,None,None]*w[None,:,None]*w[None,None,:]
        direct = cap(old,ua)*w[:,None]*w[None,:]-2*cap(post_old+born,mean)
        # Full contractions and explicit representation of repeated-index content.
        combo = contract(a,b,c,ua-2*w*mean,w)
        combo -= contract(repeated(old),eye,eye,ua,w)
        combo -= 2*(contract(x,eye,y,mean,np.ones(n))-contract(repeated(post_old+born),eye,eye,mean,np.ones(n)))
        np.fill_diagonal(combo,0)
        error = float(np.max(np.abs(combo-direct)))
        # Independent check of identity-born D3 and D21 closed forms.
        d3 = np.diag(x)*np.diag(y)
        d21 = (np.diag(x)[:,None]*y.T+x*y+np.diag(y)[:,None]*x.T)/3
        expected = np.array([[born[i,i,j] for j in range(n)] for i in range(n)])
        birth_error = float(np.max(np.abs(d21-expected)))
        diagonal_error = float(np.max(np.abs(d3-np.array([born[i,i,i] for i in range(n)]))))
        assert max(error,birth_error,diagonal_error)<1e-11
        rows.append({'n':n,'merged_capped_error':error,'birth_d21_error':birth_error,'birth_d3_error':diagonal_error})
    p=Path('results/phase2/source_birth_combination_algebra.json')
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps({'mode':'synthetic algebra only; no benchmark score','seed':7,'rows':rows,
        'formula':'cap(old,ua)*w_j*w_k - 2 cap(wicked_old+birth,mean) = merged old contraction with ua-2*w*mean plus repeated-content/birth corrections',
        'source':'upstream lean/lean_k3_aug.py F52 combined contraction; commit 1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d',
        'limits':['Verifies source bookkeeping, not complete power-cumulant diagram production','Arbitrary weights ua,w,mean; their ReLU assignment requires production formula','Dense tensor exists offline only']},indent=2)+'\n')
    print('Three birth/repeated/merged-source checks passed', rows)


if __name__=='__main__':
    main()
