"""Bounded complete official k4-basic producer and streaming K4 contraction.

Diagnostic n<=16. No references or fitting to network scores. Official full
k4-basic diagrams, with power-to-ordinary partition conversion generated here.
No charged n4 output allocation; streaming still requires O(n4) total work.
"""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"

from collections import Counter,defaultdict
from fractions import Fraction
from functools import lru_cache
import itertools
import math
from pathlib import Path
import shutil
import tempfile

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.audit_official_augmented_terms import coefficient,extract_generators


@lru_cache(None)
def setparts(size):
    if size==0:return ((),)
    out=[]
    for part in setparts(size-1):
        out.append(part+((size-1,),))
        for i in range(len(part)):
            out.append(part[:i]+(part[i]+(size-1,),)+part[i+1:])
    return tuple(out)


@lru_cache(None)
def ordinary_polynomial(powers):
    """Direct cumulant partition followed by raw-moment/power-cumulant inverse."""
    labels=tuple(j for j,p in enumerate(powers) for _ in range(p))
    out=Counter()
    for outer in setparts(len(labels)):
        # Each raw moment is itself a sum of partitions of distinct groups.
        choices=[]
        for block in outer:
            counts=Counter(labels[t] for t in block)
            groups=tuple(sorted(counts))
            poly=[]
            for inner in setparts(len(groups)):
                monomial=[]
                for ib in inner:
                    v=[0]*len(powers)
                    for t in ib:v[groups[t]]=counts[groups[t]]
                    monomial.append(tuple(v))
                poly.append(tuple(sorted(monomial)))
            choices.append(poly)
        coef=(-1)**(len(outer)-1)*math.factorial(len(outer)-1)
        for prod in itertools.product(*choices):
            out[tuple(sorted(itertools.chain.from_iterable(prod)))]+=coef
    return tuple((m,c) for m,c in sorted(out.items()) if c)


@lru_cache(None)
def official_terms():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)/"src/mlp_kprop";d.mkdir(parents=True)
        for name in ("partitions.py","kprop_harmonic.py"):
            shutil.copyfile(Path("benchmark/data/official_source_93d091a4")/name,d/name)
        ns,sources=extract_generators(Path(td))
        raw=ns["get_all_terms"](4,d_max=4,augment=False)
        result=defaultdict(list)
        for ip,vp in raw:
            result[ip].append((vp,float(coefficient(vp,1))))
        inventory={"official_commit":"93d091a4c26c042bfffa28f2e76a81bc0aba94bb",
                   "sources":sources,"k_max":4,"d_max":4,"augment":False,
                   "raw_counts":{str(k):len(v) for k,v in result.items()},
                   "1111_factor_filter_all_kept":all(ns["factored_keeps_term"](4,p,v) for p,v in raw if p==(1,1,1,1)),
                   "1111_terms":[[v,c] for v,c in result[(1,1,1,1)]],
                   "note":"k_max3 int-condition cannot produce1111; k4basic budget3 is the explicit changed closure"}
    return dict(result),inventory


def relu_wicks(mu,sigma,pairs):
    alpha=mu/sigma
    phi=flops.stats.norm.pdf(alpha).astype(fnp.float64)
    cdf=flops.stats.norm.cdf(alpha).astype(fnp.float64)
    moments={0:cdf,1:mu*cdf+sigma*phi}
    for p in range(2,5):moments[p]=mu*moments[p-1]+(p-1)*sigma*sigma*moments[p-2]
    hs={0:fnp.ones(len(mu),dtype=fnp.float64),1:alpha}
    for k in range(2,5):hs[k]=alpha*hs[k-1]-(k-1)*hs[k-2]
    result={}
    for k,p in pairs:
        if k<=p:
            result[(k,p)]=math.factorial(p)/math.factorial(p-k)*moments[p-k]
        else:
            q=k-p-1
            result[(k,p)]=math.factorial(p)*(-1)**q*hs[q]*phi/sigma**(k-p)
    return result


def pair_entry(a,b,indices):
    i,j,k,l=indices
    return (a[i,j]*b[k,l]+a[i,k]*b[j,l]+a[i,l]*b[j,k]
            +b[i,j]*a[k,l]+b[i,k]*a[j,l]+b[i,l]*a[j,k])/6


class Producer:
    """Produces arbitrary specified entries in metered arrays, no output n4."""
    def __init__(self,mu,cov,k3,pairs):
        self.mu,self.cov,self.k3,self.pairs=mu,cov,k3,pairs
        self.n=len(mu);self.terms,_=official_terms()
        self.wicks=relu_wicks(mu,fnp.sqrt(fnp.diag(cov)),
            {(sum(v[axis] for v in vp),p) for ip,terms in self.terms.items()
             for vp,_ in terms for axis,p in enumerate(ip)})

    def input_entry(self,vector,axes):
        indices=tuple(index for power,index in zip(vector,axes) for _ in range(power))
        degree=len(indices)
        if degree==2:return self.cov[indices]
        if degree==3:return self.k3[indices]
        if degree==4:
            result=fnp.zeros(np.shape(axes[0]),dtype=fnp.float64)
            for a,b in self.pairs:result=result+pair_entry(a,b,indices)
            return result
        raise ValueError(vector)

    def power(self,vector,axes):
        # Select only coordinates with nonzero power; sort powers descending.
        ids=sorted((i for i,v in enumerate(vector) if v),key=lambda i:(-vector[i],i))
        p=tuple(vector[i] for i in ids);selected=tuple(axes[i] for i in ids)
        if len(selected)==1:result=fnp.zeros(np.shape(selected[0]),dtype=fnp.float64)
        else:result=fnp.zeros(np.shape(selected[0]),dtype=fnp.float64)
        for vp,coefficient_ in self.terms[p]:
            val=fnp.ones(np.shape(selected[0]),dtype=fnp.float64)*coefficient_
            for v in vp:val=val*self.input_entry(v,selected)
            for axis,power in enumerate(p):
                val=val*self.wicks[(sum(v[axis] for v in vp),power)][selected[axis]]
            result=result+val
        return result

    def entries(self,indices):
        d=len(indices);shape=np.shape(indices[0])
        out=fnp.zeros(shape,dtype=fnp.float64)
        for blocks in setparts(d):
            # Only this exact equality pattern participates.
            mask=np.ones(shape,dtype=bool)
            owner={i:j for j,b in enumerate(blocks) for i in b}
            for i in range(d):
                for j in range(i):mask &= (indices[i]==indices[j]) if owner[i]==owner[j] else (indices[i]!=indices[j])
            if not np.any(mask):continue
            blocks=sorted(blocks,key=lambda b:(-len(b),b[0]))
            powers=tuple(map(len,blocks));axes=tuple(indices[b[0]] for b in blocks)
            cache={}
            value=fnp.zeros(shape,dtype=fnp.float64)
            for monomial,coef in ordinary_polynomial(powers):
                term=fnp.ones(shape,dtype=fnp.float64)*coef
                for v in monomial:
                    if v not in cache:cache[v]=self.power(v,axes)
                    term=term*cache[v]
                value=value+term
            out=out+value*fnp.asarray(mask,dtype=fnp.float64)
        return out

    def tensor(self,degree):
        assert degree<=3,"Charged full n4 allocation is intentionally forbidden"
        indices=tuple(np.indices((self.n,)*degree))
        return self.entries(indices)

    def row(self,i,j):
        k,l=np.indices((self.n,self.n))
        return self.entries((np.full_like(k,i),np.full_like(k,j),k,l))

    def apply(self,b):
        """(H:B)_ij, not average-Sym adjoint scaling; streaming n2 temporary."""
        return fnp.stack([fnp.stack([fnp.sum(self.row(i,j)*b) for j in range(self.n)]) for i in range(self.n)])


def inherited_row(pairs,w1,i,j,n):
    k,l=np.indices((n,n));idx=(np.full_like(k,i),np.full_like(k,j),k,l)
    value=fnp.zeros((n,n),dtype=fnp.float64)
    for a,b in pairs:
        value=value+pair_entry(a,b,idx)*w1[i]*w1[j]*w1[k]*w1[l]
    return value


def harmonic_identity_core(trace,n):
    return 6/(n+4)*trace-3/((n+2)*(n+4))*fnp.sum(fnp.diag(trace))*fnp.eye(n,dtype=fnp.float64)


def project_birth_r2(producer,basis):
    """Best identity-radial projection plus one fixed residual pair direction.

    Coefficients derive only from represented model tensor, never target scores.
    Keeps two matrix pairs after orthogonalizing S(basis,basis) against S(G,I).
    """
    n=producer.n;eye=fnp.eye(n,dtype=fnp.float64)
    w1=producer.wicks[(1,1)]
    def birthrow(i,j):return producer.row(i,j)-inherited_row(producer.pairs,w1,i,j,n)
    trace=fnp.zeros((n,n),dtype=fnp.float64)
    for i in range(n):trace=trace+birthrow(i,i)
    core=harmonic_identity_core(trace,n)
    # Trace of S(B,B) = [tr(B) B +2 B B]/3.
    qb_trace=(fnp.sum(fnp.diag(basis))*basis+2*(basis@basis))/3
    qcore=harmonic_identity_core(qb_trace,n)
    numerator=denominator=fnp.asarray(0.,dtype=fnp.float64)
    for i in range(n):
        for j in range(n):
            k,l=np.indices((n,n));idx=(np.full_like(k,i),np.full_like(k,j),k,l)
            residual=birthrow(i,j)-pair_entry(core,eye,idx)
            q=pair_entry(basis,basis,idx)-pair_entry(qcore,eye,idx)
            numerator=numerator+fnp.sum(residual*q)
            denominator=denominator+fnp.sum(q*q)
    alpha=numerator/fnp.maximum(denominator,1e-30)
    return [(core-alpha*qcore,eye),(alpha*basis,basis)],alpha
