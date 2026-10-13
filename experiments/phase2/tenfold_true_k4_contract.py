"""Two-step n4/8/16 complete retained K4 producer, streaming and fixed-R2.

No benchmark reference, candidate score or production1024 run. Dense independent
offline diagram oracle is bounded. This proves bookkeeping/representation only.
"""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from experiments.phase2.tenfold_true_k4_producer import (
    Producer,official_terms,ordinary_polynomial,setparts,project_birth_r2)


def numpy_wick(k,p,mu,sigma):
    alpha=mu/sigma
    phi=np.exp(-alpha*alpha/2)/math.sqrt(2*math.pi)
    cdf=np.array([.5*(1+math.erf(x/math.sqrt(2))) for x in alpha])
    moments={0:cdf,1:mu*cdf+sigma*phi}
    for d in range(2,5):moments[d]=mu*moments[d-1]+(d-1)*sigma*sigma*moments[d-2]
    if k<=p:return math.factorial(p)/math.factorial(p-k)*moments[p-k]
    q=k-p-1;h=np.polynomial.hermite_e.hermeval(alpha,[0]*q+[1])
    return math.factorial(p)*(-1)**q*h*phi/sigma**(k-p)


def pair_dense(a,b):
    t=np.einsum("ij,kl->ijkl",a,b)
    return sum(t.transpose(p) for p in itertools.permutations(range(4)))/24


def numpy_producer(mu,cov,k3,pairs):
    """Independent scalar diagram sum + raw-moment cumulant partitions."""
    n=len(mu);terms,_=official_terms();h=sum(pair_dense(a,b) for a,b in pairs)
    wicks={pair:numpy_wick(*pair,mu,np.sqrt(np.diag(cov))) for pair in
           {(sum(v[a] for v in vp),p) for ip,ts in terms.items() for vp,_ in ts for a,p in enumerate(ip)}}
    def power(powers,groups):
        order=sorted(range(len(powers)),key=lambda i:(-powers[i],i))
        powers=tuple(powers[i] for i in order);groups=tuple(groups[i] for i in order)
        value=0.
        for vp,coef in terms[powers]:
            t=coef
            for v in vp:
                ids=tuple(groups[a] for a,p in enumerate(v) for _ in range(p))
                t*=({2:cov,3:k3,4:h}[len(ids)])[ids]
            for a,p in enumerate(powers):t*=wicks[(sum(v[a] for v in vp),p)][groups[a]]
            value+=t
        return value
    # Raw moments by distinct coordinate powers, expanding their cumulants.
    rawcache={}
    def raw(indices):
        counts=Counter(indices);groups=tuple(sorted(counts));key=tuple(sorted(indices))
        if key in rawcache:return rawcache[key]
        value=0.
        for part in setparts(len(groups)):
            term=1.
            for block in part:
                term*=power(tuple(counts[groups[t]] for t in block),tuple(groups[t] for t in block))
            value+=term
        rawcache[key]=value;return value
    outputs=[]
    for degree in range(1,5):
        out=np.zeros((n,)*degree)
        for idx in itertools.combinations_with_replacement(range(n),degree):
            value=0.
            for part in setparts(degree):
                term=(-1)**(len(part)-1)*math.factorial(len(part)-1)
                for block in part:term*=raw(tuple(idx[i] for i in block))
                value+=term
            for perm in set(itertools.permutations(idx)):out[perm]=value
        outputs.append(out)
    return outputs


def operator_errors(before,after,tests):
    errors=[]
    for b in tests:
        u,v=np.einsum("ijkl,kl->ij",before,b),np.einsum("ijkl,kl->ij",after,b)
        errors.append(float(np.linalg.norm(u-v)/max(np.linalg.norm(u),1e-30)))
    return errors


def finite_distribution_partition_contract():
    # Separately verifies generated pK->K polynomial using an actual arbitrary
    # finite joint distribution. No Hermite approximation is tested here.
    rng=np.random.default_rng(261013970);y=rng.normal(size=(19,4));weights=np.arange(1,20,dtype=float);weights/=weights.sum()
    worst=0.
    for powers in ((4,),(3,1),(2,2),(2,1,1),(1,1,1,1)):
        def cumulant_of_vars(values):
            d=len(values);value=0.
            for part in setparts(d):
                term=(-1)**(len(part)-1)*math.factorial(len(part)-1)
                for block in part:term*=float(weights@np.prod(np.stack([values[i] for i in block]),axis=0))
                value+=term
            return value
        pks={v:cumulant_of_vars([y[:,j]**p for j,p in enumerate(v) if p])
             for monomial,_ in ordinary_polynomial(powers) for v in monomial}
        formula=sum(coef*math.prod(pks[v] for v in monomial) for monomial,coef in ordinary_polynomial(powers))
        direct=cumulant_of_vars([y[:,j] for j,p in enumerate(powers) for _ in range(p)])
        worst=max(worst,abs(formula-direct))
    assert worst<1e-12
    return worst


def run(n):
    started=time.monotonic();rng=np.random.default_rng(261013971+n)
    raw=rng.normal(size=(n,n));cov=.05*(raw@raw.T/n)+np.eye(n)
    mu=rng.normal(size=n)*.2
    u=rng.normal(size=(n,2))*.03
    k3=np.einsum("ir,jr,kr->ijk",u,u,u)
    a=np.diag(rng.normal(size=n)*.025);b=np.eye(n)+cov*.1
    pairs=[(a,b)];tests=[np.eye(n),cov,(lambda z:(z+z.T)/2)(rng.normal(size=(n,n)))]
    weight=np.eye(n)*.9+rng.normal(size=(n,n))*.04/math.sqrt(n)
    costs=[];steps=[]
    for step in range(2):
        print(json.dumps({"width":n,"step":step,"phase":"independent_scalar_oracle"}),flush=True)
        oracle=numpy_producer(mu,cov,k3,pairs)
        print(json.dumps({"width":n,"step":step,"phase":"charged_complete_producer"}),flush=True)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            fm,fc,fk=[fnp.asarray(x) for x in (mu,cov,k3)]
            fp=[(fnp.asarray(a),fnp.asarray(b)) for a,b in pairs]
            producer=Producer(fm,fc,fk,fp)
            model=[producer.tensor(d) for d in (1,2,3)]
            # Complete streaming component creates <=n2 H4 row temporaries.
            applied=[producer.apply(fnp.asarray(t)) for t in tests]
            projected,alpha=project_birth_r2(producer,fc)
        costs.append(ctx.summary_dict())
        modelnp=[np.asarray(x).copy() for x in model]
        deltas=[float(np.max(abs(x-y))) for x,y in zip(modelnp,oracle[:3])]
        operator_delta=max(float(np.max(abs(np.asarray(x)-np.einsum("ijkl,kl->ij",oracle[3],t)))) for x,t in zip(applied,tests))
        assert max(deltas+[operator_delta])<3e-9,(n,step,deltas,operator_delta)
        inherited=sum(pair_dense(a,b) for a,b in pairs)
        gain=np.asarray(producer.wicks[(1,1)])
        inherited=inherited*gain[:,None,None,None]*gain[None,:,None,None]*gain[None,None,:,None]*gain[None,None,None,:]
        birth=oracle[3]-inherited
        projnp=[(np.asarray(a).copy(),np.asarray(b).copy()) for a,b in projected]
        approx=sum(pair_dense(a,b) for a,b in projnp)
        trace=np.einsum("iijk->jk",birth)
        g=6/(n+4)*trace-3/((n+2)*(n+4))*np.trace(trace)*np.eye(n)
        r1=pair_dense(g,np.eye(n))
        steps.append({"step":step,"mean_cov_K3_scalar_oracle_maxdeltas":deltas,
                      "complete_K4_streaming_apply_scalar_oracle_maxdelta":operator_delta,
                      "birth_tensor_norm":float(np.linalg.norm(birth)),
                      "birth_before_r1_relative_Frobenius":float(np.linalg.norm(birth-r1)/np.linalg.norm(birth)),
                      "birth_after_r2_relative_Frobenius":float(np.linalg.norm(birth-approx)/np.linalg.norm(birth)),
                      "birth_before_r1_operator_relative_errors":operator_errors(birth,r1,tests),
                      "birth_after_r2_operator_relative_errors":operator_errors(birth,approx,tests),
                      "nonoracle_fixed_basis_scalar_alpha":float(alpha),
                      "actual_inherited_tensor_norm":float(np.linalg.norm(inherited)),
                      "legal_cost":ctx.summary_dict()})
        assert np.linalg.norm(birth-approx)<=np.linalg.norm(birth-r1)+1e-9
        # Keep actual inherited pair + two birth pairs. No implicit history loss.
        totalpairs=[(a*gain[:,None]*gain[None,:],b*gain[:,None]*gain[None,:]) for a,b in pairs]+projnp
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            fw=fnp.asarray(weight)
            moved=[(fw@fnp.asarray(a)@fw.T,fw@fnp.asarray(b)@fw.T) for a,b in totalpairs]
            nextmu=fw@model[0];nextcov=fw@model[1]@fw.T
            nextk3=fnp.einsum("ia,jb,kc,abc->ijk",fw,fw,fw,model[2])
        transported_pairs=[(np.asarray(a).copy(),np.asarray(b).copy()) for a,b in moved]
        explicit=np.einsum("ia,jb,kc,ld,abcd->ijkl",weight,weight,weight,weight,inherited+approx,optimize=True)
        factortransport=sum(pair_dense(a,b) for a,b in transported_pairs)
        transportdelta=float(np.max(abs(explicit-factortransport)))
        assert transportdelta<1e-9
        steps[-1].update({"actual_W_pair_history_transport_maxdelta":transportdelta,"history_pairs_after_birth":len(totalpairs),
                          "transport_legal_cost":ctx.summary_dict()})
        mu,cov,k3=[np.asarray(x).copy() for x in (nextmu,nextcov,nextk3)]
        # Numerically near-singular toy C may need no repair; report exact range.
        assert np.min(np.linalg.eigvalsh((cov+cov.T)/2))>0, {"width":n,"step":step,"mincov":float(np.min(np.linalg.eigvalsh((cov+cov.T)/2)))}
        pairs=transported_pairs
    return {"width":n,"steps":steps,"wall_time_s":time.monotonic()-started,
            "maximum_charged_K4_temporary_elements":n**3,"dense_K4_offline_only_elements":n**4,
            "scaling":"streaming n^2 rows*n^2 entries*constant diagrams: O(n^4) work; no production1024 claim"}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--widths",type=int,nargs="+",default=[4,8,16]);args=p.parse_args();assert not args.output.exists()
    assert all(n<=16 for n in args.widths)
    _,inventory=official_terms();rows=[]
    for n in args.widths:
        row=run(n);rows.append(row);print(json.dumps({"width":n,"wall":row["wall_time_s"],"step_costs":[s["legal_cost"]["flops_used"] for s in row["steps"]],"before_after_r1_r2":[[s["birth_before_r1_relative_Frobenius"],s["birth_after_r2_relative_Frobenius"]] for s in row["steps"]]}),flush=True)
    report={"version":"tenfold-true-k4-producer-contract-r04","complete":True,"inventory":inventory,"rows":rows,
            "independent_finite_distribution_pK_to_K_maxerror":finite_distribution_partition_contract(),
            "source_sha256":{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (Path(__file__),Path("experiments/phase2/tenfold_true_k4_producer.py"))},
            "before":"single identity harmonic birth approximation",
            "after":"full k4basic retained streaming producer, subtract actual inherited fourgain pair, identity radial+one orthogonal fixed covariance-square birth direction; preserve history pairs",
            "no_adjusted_score":True,"not_resource_acceptance":True,
            "limitations":["true in filename distinguishes full retained producer from old proxy, not exact true distribution",
                           "Hermite diagram truncation and fixed R2 compression remain; R2 per birth, histories grow1->3->5, not globally fixed2",
                           "Independent oracle shares pinned diagram specification but separate scalar rawmoment cumulant evaluation; finite sample validates only pK-to-K",
                           "Streaming computation O(n4) work, n3 K3 toy transport; not scalable production1024",
                           "No realbenchmark reference, raw/adjusted or remote resource run"]}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":main()
