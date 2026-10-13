"""Charged full-metric H4 slices/caps and all three K4->K3 feed groups.

No network targets, references, estimator changes or score claims. Dense
4/3-tensors occur only in the independent small-shape offline oracle.
"""
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def slices(a, b):
    """Average-Sym4(A tensor B); matrices symmetric, including diagonal."""
    ad, bd = fnp.diag(a), fnp.diag(b)
    d4 = ad*bd
    d31 = (ad[:, None]*b+bd[:, None]*a)/2
    d22 = (ad[:, None]*bd[None, :]+bd[:, None]*ad[None, :]+4*a*b)/6
    return d4, d31, d22


def weighted_distinct_211(a, b, z):
    """F[j,k]=sum distinct i z[i] H[i,i,j,k]; diagonal output zero."""
    _, d31, _ = slices(a, b)
    value = (fnp.sum(z*fnp.diag(a))*b+fnp.sum(z*fnp.diag(b))*a
             +2*((a*z[None, :])@b+(b*z[None, :])@a))/6
    value = value-z[:, None]*d31-z[None, :]*d31.T
    fnp.fill_diagonal(value, 0)
    return value


def weighted_distinct_121(a, b, z):
    """P[j,k]=sum distinct i z[i] H[i,j,j,k]; diagonal output zero."""
    _, d31, d22 = slices(a, b)
    az, bz = a@z, b@z
    value = (fnp.diag(a)[:, None]*bz[None, :]
             +fnp.diag(b)[:, None]*az[None, :]
             +2*b*az[:, None]+2*a*bz[:, None])/6
    value = value-z[:, None]*d31-z[None, :]*d22
    fnp.fill_diagonal(value, 0)
    return value


def feed_groups(a, b, w1, w2):
    """Official factor_k3.py's complete three groups, now arbitrary A/B."""
    n = len(w1)
    eye, ones = fnp.eye(n, dtype=a.dtype), fnp.ones((n,n), dtype=a.dtype)
    return [
        (w2[:, None]*fnp.diag(a)[:, None]*ones,
         w1[:, None]*b, w1[:, None]*eye/4),
        (w1[:, None]*a, w2[:, None]*eye, w1[:, None]*b),
        (w1[:, None]*a, w1[:, None]*eye,
         w2[:, None]*fnp.diag(b)[:, None]*ones/4),
    ]


def direct_feed_slices_and_center(a,b,w1,w2,mean):
    """Same three-group tensor via H4 slices, no CP tensor contractions."""
    d4,d31,d22=slices(a,b)
    d3=1.5*w2*w1*w1*d4
    d21=w2[:,None]*w1[:,None]*w1[None,:]*d31
    d21=d21+0.5*(w1*w1)[:,None]*w2[None,:]*d22
    fnp.fill_diagonal(d21,0)
    center=weighted_distinct_211(a,b,mean*w2)*w1[:,None]*w1[None,:]
    near=weighted_distinct_121(a,b,mean*w1)
    center=center+near*w2[:,None]*w1[None,:]+near.T*w1[:,None]*w2[None,:]
    return d3,d21,-center


def cp_slices(x, y, z):
    d3 = fnp.sum(x*y*z, axis=1)
    d21 = ((x*y)@z.T+(x*z)@y.T+(y*z)@x.T)/3
    fnp.fill_diagonal(d21, 0)
    return d3, d21


def cp_weighted_distinct(x, y, z, weights):
    """sum distinct i weights[i] averageSym3 CP[i,j,k]."""
    _, d21 = cp_slices(x, y, z)
    sx, sy, sz = x.T@weights, y.T@weights, z.T@weights
    full = ((y*sx[None, :])@z.T+(z*sy[None, :])@x.T
            +(x*sz[None, :])@y.T)/3
    full = (full+full.T)/2
    full = full-weights[:, None]*d21-weights[None, :]*d21.T
    fnp.fill_diagonal(full, 0)
    return full


def sym_dense(arr):
    return sum(arr.transpose(p) for p in itertools.permutations(range(arr.ndim)))/math.factorial(arr.ndim)


def dense_h(a, b):
    return sym_dense(np.einsum("ij,kl->ijkl", a, b))


def dense_cp(groups):
    return sum(sym_dense(np.einsum("ir,jr,kr->ijk", *(np.asarray(x) for x in group))) for group in groups)


def check(n, rng, diagonal_metric=False):
    a, b = [rng.normal(size=(n,n)) for _ in range(2)]
    a, b = (a+a.T)/2, (b+b.T)/2
    if diagonal_metric:
        b = np.diag(np.diag(b))
    w, gain = rng.normal(size=(n,n)), rng.normal(size=n)
    w1, w2, w12, w22, mean = rng.normal(size=(5,n))
    h = dense_h(a,b)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
        fa, fb, fw, fg = [fnp.asarray(x) for x in (a,b,w,gain)]
        d4, d31, d22 = slices(fa,fb)
        fw1, fw2, fw12, fw22, fm = [fnp.asarray(x) for x in (w1,w2,w12,w22,mean)]
        fcap = weighted_distinct_211(fa,fb,fw22)
        pcap = weighted_distinct_121(fa,fb,fw12)
        raw_cap = (fcap*fw1[:,None]*fw1[None,:]
                   +pcap*fw2[:,None]*fw1[None,:]
                   +pcap.T*fw1[:,None]*fw2[None,:])/2
        groups = feed_groups(fa,fb,fw1,fw2)
        born = [cp_slices(*g) for g in groups]
        feed_center = sum(cp_weighted_distinct(*g,fm) for g in groups)*(-2)
        direct_feed=direct_feed_slices_and_center(fa,fb,fw1,fw2,fm)
        mapped = fw*fg[None,:]
        transported_a, transported_b = mapped@fa@mapped.T, mapped@fb@mapped.T
        transported_slices = slices(transported_a,transported_b)
        transported_groups = [tuple(fw@x for x in g) for g in groups]
        moved = [cp_slices(*g) for g in transported_groups]
    legal_cost = context.summary_dict()
    expected_cap, expected_feed = np.zeros((n,n)), np.zeros((n,n))
    f = dense_cp(groups)
    for j,k in itertools.product(range(n),repeat=2):
        if j==k:
            continue
        for i in range(n):
            if i not in (j,k):
                expected_cap[j,k] += (h[i,i,j,k]*w22[i]*w1[j]*w1[k]
                                     +h[i,j,j,k]*w12[i]*w2[j]*w1[k]
                                     +h[i,j,k,k]*w12[i]*w1[j]*w2[k])/2
                expected_feed[j,k] -= 2*mean[i]*f[i,j,k]
    mapped_np = w*gain[None,:]
    th = np.einsum("ia,jb,kc,ld,abcd->ijkl",mapped_np,mapped_np,mapped_np,mapped_np,h,optimize=True)
    tf = np.einsum("ia,jb,kc,abc->ijk",w,w,w,f,optimize=True)
    expected21, transported21 = np.einsum("iij->ij",f).copy(),np.einsum("iij->ij",tf).copy()
    np.fill_diagonal(expected21,0);np.fill_diagonal(transported21,0)
    errors = {
        "D4":float(np.max(abs(np.asarray(d4)-np.einsum("iiii->i",h)))),
        "D31":float(np.max(abs(np.asarray(d31)-np.einsum("iiij->ij",h)))),
        "D22":float(np.max(abs(np.asarray(d22)-np.einsum("iijj->ij",h)))),
        "all_three_raw_power_211_diagrams":float(np.max(abs(np.asarray(raw_cap)-expected_cap))),
        "all_three_feed_groups_D3":float(np.max(abs(sum(np.asarray(x[0]) for x in born)-np.einsum("iii->i",f)))),
        "all_three_feed_groups_D21":float(np.max(abs(sum(np.asarray(x[1]) for x in born)-expected21))),
        "feed_distinct_mean_center":float(np.max(abs(np.asarray(feed_center)-expected_feed))),
        "direct_feed_D3":float(np.max(abs(np.asarray(direct_feed[0])-np.einsum("iii->i",f)))),
        "direct_feed_D21":float(np.max(abs(np.asarray(direct_feed[1])-expected21))),
        "direct_feed_distinct_mean_center":float(np.max(abs(np.asarray(direct_feed[2])-expected_feed))),
        "transport_D4":float(np.max(abs(np.asarray(transported_slices[0])-np.einsum("iiii->i",th)))),
        "transport_D31":float(np.max(abs(np.asarray(transported_slices[1])-np.einsum("iiij->ij",th)))),
        "transport_D22":float(np.max(abs(np.asarray(transported_slices[2])-np.einsum("iijj->ij",th)))),
        "feed_transport_D3":float(np.max(abs(sum(np.asarray(x[0]) for x in moved)-np.einsum("iii->i",tf)))),
        "feed_transport_D21":float(np.max(abs(sum(np.asarray(x[1]) for x in moved)-transported21))),
    }
    assert max(errors.values()) < 1e-10, errors
    return {"width":n,"metric":"diagonal" if diagonal_metric else "dense","max_abs_errors":errors,
            "legal_component_cost":legal_cost,
            "scope":"float64 arbitrary symmetric A/B incl diagonal, signed gains, all feed groups and actual W/WD transport; no parent/score"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    rng=np.random.default_rng(261013951)
    rows=[check(n,rng,diagonal) for n in (4,8) for diagonal in (False,True)]
    report={"version":"tenfold-full-k4-component-contract-r02","complete":True,"rows":rows,
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "normalization":"Sym4 averages 24 permutations; symmetric A/B yields six distinct terms each 1/6; CP3 averages6",
            "no_adjusted_score":True,"not_resource_acceptance":True,
            "limitations":["Charged generic CP helpers intentionally unoptimized; not 1024 cost projection",
                           "Exact contractions of represented sum, not true nonlinear K4 closure or production integration",
                           "No pruning/confine/parent residual dedup contract yet"]}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"rows":[{"width":r["width"],"metric":r["metric"],"max_abs_error":max(r["max_abs_errors"].values()),"flops":r["legal_component_cost"]["flops_used"]} for r in rows]},indent=2))


if __name__=="__main__":
    main()
