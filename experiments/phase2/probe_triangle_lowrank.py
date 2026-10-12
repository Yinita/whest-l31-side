"""Independent offline checks for an all-distinct covariance triangle.

Checks coefficient using exact Isserlis pairings for centered squares, then
compares the rank-r contraction with explicitly materialized small tensors.
"""
import itertools
import json
from pathlib import Path

import numpy as np


def pairings(items):
    if not items:
        yield []
        return
    first=items[0]
    for j in range(1,len(items)):
        for rest in pairings(items[1:j]+items[j+1:]):
            yield [(first,items[j])]+rest


def gaussian_moment(c,indices):
    return sum(np.prod([c[i,j] for i,j in p]) for p in pairings(list(indices)))


def fast_contraction(q,b,w,w2):
    n,r=q.shape
    right=q@b
    features=(q[:,:,None]*q[:,None,:]).reshape(n,r*r)
    right_features=(right[:,:,None]*right[:,None,:]).reshape(n,r*r)
    diagonal=np.sum(q*right,axis=1)
    a=w*w2[None,:]
    m=(a@features).reshape(n,r,r)
    k=b[None,:,:]@m
    full=np.sum((k@k)*k.transpose(0,2,1),axis=(1,2))
    two_equal=np.sum(((a*a*diagonal[None,:])@features)*(a@right_features),axis=1)
    three_equal=np.sum((a*diagonal[None,:])**3,axis=1)
    return full-3*two_equal+2*three_equal


def main():
    output=Path('results/phase2/triangle_lowrank_algebra.json')
    if output.exists():
        raise FileExistsError(output)
    rng=np.random.default_rng(279)
    quadratic_rows=[]
    for _ in range(4):
        x=rng.normal(size=(3,3))
        c=x@x.T
        moment=gaussian_moment(c,[0,0,1,1,2,2])
        centered=moment-sum(c[i,i]*gaussian_moment(c,[j,j,k,k])
                            for i,j,k in ((0,1,2),(1,0,2),(2,0,1)))+2*np.prod(np.diag(c))
        expected=8*c[0,1]*c[1,2]*c[2,0]
        assert abs(centered-expected)<1e-10
        quadratic_rows.append(dict(centered_squares_cumulant=centered,
                                   triangle_wick2_product=expected,abs_error=abs(centered-expected)))
    rows=[]
    for n in (4,6,9):
        for rank in (2,n):
            q,_=np.linalg.qr(rng.normal(size=(n,rank)))
            b=rng.normal(size=(rank,rank))
            b=(b+b.T)/2
            c=q@b@q.T
            w=rng.normal(size=(n,n))
            w2=rng.normal(size=n)
            t=np.zeros((n,n,n))
            for i,j,k in itertools.product(range(n),repeat=3):
                if len({i,j,k})==3:
                    t[i,j,k]=w2[i]*w2[j]*w2[k]*c[i,j]*c[j,k]*c[k,i]
            explicit=np.einsum('ijk,ai,aj,ak->a',t,w,w,w,optimize=True)
            fast=fast_contraction(q,b,w,w2)
            error=float(np.max(np.abs(fast-explicit)))
            assert error<3e-11
            rows.append(dict(n=n,rank=rank,contraction_max_error=error,
                             repeats_zero=True,diagonal_is_not_zero=float(np.linalg.norm(np.diag(c)))))
    report=dict(mode='offline small synthetic correctness; no score improvement',seed=279,
                formula='Tijk=w2_i*w2_j*w2_k*Cij*Cjk*Cki for three distinct indices; zero otherwise',
                coefficient='One undirected covariance triangle, coefficient 1 (3!/(3!) from edge-vector partition)',
                formal_grade='Basic algorithm equation14: 1+3*(2-1-0)=4 for edges (1,1,0),(1,0,1),(0,1,1); total Hermite order6 exceeds equation13 K3 limit5. Current V29 factored K111 sources omit the triangle; custom augmented term selection may differ',
                exact_dense_cost='One trace((diag(a)C)**3) per output; O(n**4) for all n outputs',
                approximation='Ctilde=Q B Q^T; indefinite symmetric low-rank projection; exact distinct mask on approximation',
                fast_formula='raw=tr((B Q^T diag(a) Q)^3); correction=raw-3sum_ik a_i²a_k d_i Ctilde_ik²+2sum_i a_i³d_i³',
                cost='Three n-by-n-by-r² matmuls plus O(n*r³); features O(n*r²), no n³ tensor',
                candidate_basis='QR of C_off @ (C_off @ contiguous next-weight sketch), rank16; deterministic, no setup randomness',
                primary_source='https://arxiv.org/html/2605.05179v2',
                sections=['S.3.2 connected diagram combinatorics','S.3.4 ReLU derivatives','S.4.1 equation14 truncation'],
                centered_square_isserlis_checks=quadratic_rows,
                explicit_tensor_checks=rows,
                limitations=['Low-rank C_off can lose substantial triangle content; exact algebra is only for projected C',
                             'Final-only one-layer marginal power correction does not transport triangle through entire depth',
                             'No truth or target-dependent sketch; network weights determine sketch'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 4 independent Isserlis coefficient checks and 6 explicit low-rank tensor contractions')
    print('Max contraction error:',max(row['contraction_max_error'] for row in rows))


if __name__=='__main__':
    main()
