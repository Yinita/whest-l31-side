"""Deterministic finite-distribution tensor-free projection/merge contract."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from joint_direction_stats_r58 import block, empty, merge, cumulants, joint_targets


def exact_cumulants(values):
    mean=values.mean(axis=0); c=values-mean; var=np.mean(c*c,axis=0)
    return np.stack([mean,var,np.mean(c**3,axis=0),np.mean(c**4,axis=0)-3*var**2],axis=-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    # Independent unequal discrete components: Cartesian product makes all
    # cross cumulants analytically zero; a correlated alternative is nonzero.
    grid=np.stack(np.meshgrid(np.array([-2.,1.,3.]),np.array([-1.,2.]),
                             np.array([0.,4.]),indexing='ij'),axis=-1).reshape(-1,3)
    directions=np.array([[1.,-.5],[.25,1.25],[-.75,.4]])
    rows=[]
    for name,x in [('independent_cartesian',grid),('shared_nonlinear',np.stack([grid[:,0],grid[:,0]**2,grid[:,0]+grid[:,1]],axis=1))]:
        projected=x@directions
        marginal=exact_cumulants(x); scalar=exact_cumulants(projected)
        diag=np.array([[sum(directions[i,j]**p*marginal[i,p-1] for i in range(3))
                       for p in range(1,5)] for j in range(2)])
        joint=scalar-diag
        if name.startswith('independent'):assert np.max(abs(joint[:,1:]))<1e-11
        else:assert np.max(abs(joint[:,1:]))>1
        packed=np.concatenate([x,projected,projected],axis=1)
        full=block(packed); accumulated=empty(packed.shape[1:])
        for part in np.array_split(packed,5):accumulated=merge(accumulated,block(part))
        merge_error=max(float(np.max(abs(accumulated[k]-full[k]))) for k in ('mean','m2','m3','m4'))
        assert merge_error<1e-8
        kfull=cumulants(full);kmerged=cumulants(accumulated);assert np.max(abs(kfull-kmerged))<1e-10
        # Permute previous-neuron axes and matching W; canonical directions stay.
        perm=np.array([2,0,1]);assert np.array_equal(x[:,perm]@directions[perm],projected)
        batch=block(packed[:,None,:]);targets=joint_targets(batch,directions[None],3)
        assert np.max(abs(targets['joint_pre4'][...,0]))<1e-12
        # k-statistics alter finite-sample cross values, so independence check
        # above uses exact population distribution, not unbiased sample k4.
        rows.append(dict(name=name,exact_joint_pre4=joint.tolist(),merge_maxabs=merge_error,
                         kstat_merge_maxabs=float(np.max(abs(kfull-kmerged))),canonical_direction_invariant=True))
    # Large offset: centered updates remain stable unlike raw-power subtraction.
    rng=np.random.default_rng(261058);z=1e6+rng.normal(size=(4096,4));full=block(z);s=empty((4,))
    for part in np.array_split(z,16):s=merge(s,block(part))
    stable_error=float(np.max(abs(cumulants(s)-cumulants(full))));assert stable_error<2e-8
    report=dict(complete=True,version='r58-finitepopulation-jointprojection-merge-contract',rows=rows,
                large_offset_kstat_error=stable_error,independent_population_joint_zero=True,
                correlated_population_joint_nonzero=True,full_tensor_not_constructed=True,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path('experiments/phase2/joint_direction_stats_r58.py'))},
                no_cloud=True,no_realnetwork_prediction=True,no_labels=True,no_fitting=True,no_scoreclaim=True,
                scope='finite-distribution mathematical direction/merge contract; no MCsignal/wholecandidate/strict')
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
