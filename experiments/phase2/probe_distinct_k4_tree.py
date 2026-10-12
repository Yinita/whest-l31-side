"""Offline exact algebra for a newborn Gaussian K4 tree residual.

The tractable object is the full tree polynomial with zero-diagonal C, minus
its identity-metric trace projection. Strict all-distinct masking introduces
additional contractions; that object is diagnosed but not called cubic cost.
Full n**4 tensors below are independent small-shape oracles only.
"""
import itertools
import json
from pathlib import Path

import numpy as np

PATHS = [p for p in itertools.permutations(range(4)) if p < p[::-1]]


def tree_tensor(c, f, h, g):
    n = len(c)
    result = np.zeros((n,n,n,n))
    for index in itertools.product(range(n), repeat=4):
        total = 0.0
        for center in range(4):
            v = index[center]
            term = g[v]
            for leaf in range(4):
                if leaf != center:
                    term *= f[index[leaf]] * c[v,index[leaf]]
            total += term
        for path in PATHS:
            i,j,k,last = (index[t] for t in path)
            total += f[i]*h[j]*h[k]*f[last]*c[i,j]*c[j,k]*c[k,last]
        result[index] = total
    return result


def raw_marginal(c, f, h, g, w):
    u = (w*f[None,:]) @ c
    star = 4*np.sum(w*g[None,:]*u**3,axis=1)
    d = w*h[None,:]*u
    chain = 12*np.sum((d@c)*d,axis=1)
    return star+chain


def trace_core(c, f, h, g):
    c2 = c*c
    s = c2@(f*f)
    one = c*g[:,None]*f[None,:]*s[:,None]
    star = one+one.T
    a = (c*(f*h)[None,:])@c2
    b = (c*(f*f)[None,:])@c
    one = f[:,None]*h[None,:]*a
    chain = 2*(one+one.T+h[:,None]*h[None,:]*c*b)
    return star+chain


def lift_core(g):
    eye = np.eye(len(g))
    return sum(np.einsum(spec,g,eye) for spec in (
        'ij,kl->ijkl','ik,jl->ijkl','il,jk->ijkl',
        'jk,il->ijkl','jl,ik->ijkl','kl,ij->ijkl'))/6


def main():
    output=Path('results/phase2/distinct_k4_tree_algebra.json')
    if output.exists():
        raise FileExistsError(output)
    rng=np.random.default_rng(735)
    rows=[]
    for n in (4,6):
        for case in range(3):
            r=rng.normal(size=(n,n))
            c=(r+r.T)/2
            np.fill_diagonal(c,0)
            f,h,g=rng.normal(size=(3,n))
            w=rng.normal(size=(n,n))
            tensor=tree_tensor(c,f,h,g)
            explicit_h=np.einsum('ijkk->ij',tensor)
            hfast=trace_core(c,f,h,g)
            gcore=6*hfast/(n+4)-3*np.trace(hfast)*np.eye(n)/((n+2)*(n+4))
            residual=tensor-lift_core(gcore)
            tensor_marginal=np.einsum('ijkl,ai,aj,ak,al->a',tensor,w,w,w,w,optimize=True)
            residual_marginal=np.einsum('ijkl,ai,aj,ak,al->a',residual,w,w,w,w,optimize=True)
            raw_fast=raw_marginal(c,f,h,g,w)
            correction=np.sum((w@gcore)*w,axis=1)*np.sum(w*w,axis=1)
            residual_fast=raw_fast-correction
            distinct=tensor.copy()
            for indices in itertools.product(range(n),repeat=4):
                if len(set(indices))!=4:
                    distinct[indices]=0
            # Distinct masking differs from harmonic subtraction even though
            # both tensors are exactly traceless.
            triangle=np.einsum('ai,aj,ak,ij,jk,ki->a',
                               (w*f[None,:])**2,w*h[None,:],w*h[None,:],c,c,c,
                               optimize=True)
            errors=dict(trace_max_error=float(np.max(np.abs(hfast-explicit_h))),
                        raw_marginal_max_error=float(np.max(np.abs(raw_fast-tensor_marginal))),
                        residual_marginal_max_error=float(np.max(np.abs(residual_fast-residual_marginal))),
                        residual_trace_max_error=float(np.max(np.abs(np.einsum('ijkk->ij',residual)))),
                        distinct_trace_max_error=float(np.max(np.abs(np.einsum('ijkk->ij',distinct)))))
            assert max(errors.values())<2e-10
            rows.append(dict(n=n,case=case,**errors,
                             distinct_vs_harmonic_residual_norm=float(np.linalg.norm(distinct-residual)),
                             all_distinct_leaf_collision_triangle_norm=float(np.linalg.norm(triangle))))
    report=dict(mode='small synthetic exact tree-polynomial algebra; no accuracy or competition evidence',
                seed=735,tree_topologies={'stars':4,'undirected_paths':len(PATHS)},
                target='full Gaussian covariance tree polynomial minus its exact h0/h2 identity-metric trace projection',
                formulas={'raw':'U=(W diag(f))C; D=(W diag(h))*U; raw=4 rowsum(W diag(g)*U**3)+12 rowsum((DC)*D)',
                          'trace':'s=(C*C)(f*f); A=(C diag(f*h))(C*C); B=(C diag(f*f))C; H=C*(g f^T diag(s)+transpose)+2[(f h^T)*A+transpose+(h h^T)*C*B]',
                          'core':'G=6H/(n+4)-3tr(H)I/((n+2)(n+4))',
                          'residual_marginal':'raw-rowsum((WG)*W)*rowsum(W*W)'},
                cost={'dense_matmuls_per_layer':5,'dense_tensor_materialization_in_candidate':False,
                      'persistent_newborn_state':'C plus f/h/g; or cached G and C plus f/h/g',
                      'implementation_ops':'matmul, multiply, add, power, sum, fill_diagonal; all flopscope-supported'},
                primary_source='https://arxiv.org/html/2605.05179v2',
                source_sections=['S.3.2 connected diagram formula','S.3.4 ReLU Hermite coefficients','S.4.2 harmonic content tradeoffs'],
                checks=rows,
                limitations=['Repeated-index entries of the full tree polynomial are a designed closure, not the exact ReLU fourth cumulant',
                             'Gaussian weak-correlation newborn production; propagated nonGaussian K3/K4 interactions omitted',
                             'Residual survives only one linear transport in proposed candidate',
                             'Strict all-distinct tensor is traceless too but exact leaf-collision correction generally requires O(n**4) batched triangle contractions',
                             'No claim that this residual has beneficial sign or size on benchmark networks'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 6 independent n**4 tensor checks for raw, trace, harmonic subtraction and one-step transport')
    print('Max error:',max(max(row[k] for k in row if k.endswith('_error')) for row in rows))


if __name__=='__main__':
    main()
