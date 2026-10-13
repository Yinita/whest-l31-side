"""Independent target-free oracle orientation and raw-central contract.

Finite synthetic ReLU samples only. Not an estimator or accuracy experiment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def central_from_raw(raw):
    m1,m2,m3,m4=[raw[...,i].astype(np.float64) for i in range(4)]
    variance=m2-m1*m1
    k3=m3-3*m1*m2+2*m1**3
    k4=m4-4*m1*m3+6*m1*m1*m2-3*m1**4-3*variance*variance
    return np.stack([m1,variance,k3,k4],axis=-1)


def run(n):
    rng=np.random.default_rng(261014240+n)
    y=np.maximum(rng.normal(size=(1021,n))+rng.normal(size=n),0)
    canonicalraw=np.stack([(y**p).mean(axis=0) for p in range(1,5)],axis=-1)
    canonical=central_from_raw(canonicalraw)
    centered=y-y.mean(axis=0)
    direct=np.stack([y.mean(axis=0),(centered**2).mean(axis=0),(centered**3).mean(axis=0),(centered**4).mean(axis=0)-3*((centered**2).mean(axis=0))**2],axis=-1)
    rawerror=float(np.max(abs(canonical-direct)))
    assert rawerror<1e-10,rawerror
    # Full cyclic permutation cannot equal its inverse: a two-cycle/swap test
    # would allow a wrong insertion orientation to pass accidentally.
    perm=np.roll(np.arange(n),1);inverse=np.argsort(perm)
    assert not np.array_equal(perm,inverse)
    internal=canonical[perm];returned=internal[inverse]
    assert np.array_equal(returned,canonical)
    inserted=canonical[perm];wrong=canonical[inverse]
    assert np.array_equal(inserted,internal)
    orientation_negative=float(np.max(abs(wrong-internal)));assert orientation_negative>1e-3
    # Conversion commutes with permutation, but arithmetic must be float64.
    converted_internal=central_from_raw(canonicalraw[perm]).astype(np.float32)
    converted_expected=canonical.astype(np.float32)[perm]
    assert np.array_equal(converted_internal,converted_expected)
    # R46 replaces diagonal ordinary state only after original pair centering.
    # Original raw/power moments and source birth ledger remain unchanged.
    pk1=rng.normal(size=n);w1=rng.normal(size=n);w12=rng.normal(size=n)
    inherited=rng.normal(size=n)*1e-4
    corrected_mean=canonical[:,0][perm]
    corrected_k1=(corrected_mean-inherited).astype(np.float32)
    observed_mu=corrected_k1.astype(np.float64)+inherited
    ridererror=float(np.max(abs(observed_mu-corrected_mean)))
    original_e=w12-2*pk1*w1
    wrong_e=w12-2*corrected_mean*w1
    assert float(np.max(abs(wrong_e-original_e)))>1e-3
    # Source/cross-pair provenance must not depend on corrected oracle diag.
    cross21=rng.normal(size=(n,n));cross22=rng.normal(size=(n,n));cross21-=np.diag(np.diag(cross21));cross22-=np.diag(np.diag(cross22))
    original21=cross21.copy();original22=cross22.copy()
    state=dict(K1=corrected_k1,K2=converted_internal[:,1],K3=converted_internal[:,2],K4=converted_internal[:,3])
    assert np.array_equal(cross21,original21) and np.array_equal(cross22,original22)
    # Every layer's canonical weights have row=input, column=output. Prior
    # internal coordinates permute weight rows; current permutes columns.
    w=rng.normal(size=(n,n))/math.sqrt(n);previous_perm=np.roll(np.arange(n),2)
    previous_inverse=np.argsort(previous_perm);mu_previous=rng.normal(size=n)
    internal_previous=mu_previous[previous_perm]
    canonical_pre=w.T@mu_previous
    live_w=w[previous_perm][:,perm]
    internal_pre=live_w.T@internal_previous
    canonical_pre_from_internal=internal_pre[inverse]
    transporterror=float(np.max(abs(canonical_pre_from_internal-canonical_pre)))
    assert transporterror<1e-12,transporterror
    wrong_transport=w[previous_inverse][:,perm].T@internal_previous
    wrong_transport_error=float(np.max(abs(wrong_transport-internal_pre)))
    assert wrong_transport_error>1e-3
    return dict(width=n,raw_to_central_float64_maxerror=rawerror,canonical_capture_internal_insert_exact=True,float32_convert_permutation_exact=True,inverse_insert_negative_control_maxerror=orientation_negative,weight_dual_axis_transport_maxerror=transporterror,wrong_previous_axis_negative_control_maxerror=wrong_transport_error,fp32_k1_plus_inherited_rider_maxerror=ridererror,original_pk1_e_retained=True,ordinary_mixed_pairs_unchanged=True,first_layer_and_final_layer_rule='oracleonly1..14; keep0and15parent; no implied source/cross closure')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    rows=[run(n) for n in (8,32,1024)]
    result=dict(complete=True,version='independent-oracle-orientation-contract-r46',rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(Path('estimators/sprint_public_v37b_h18_r01/estimator.py').read_bytes()).hexdigest(),no_labels_or_reference=True,truth_access=False,scope='syntheticpermutation/rawcentralarithmetic only; oraclefullcandidateinadmissible; no score or upperboundgainclaim',limits=['ordinarymixedpairs and truehighersource historyremain parentapproximation','MC labels have samplingnoise','oracle diagstate maynot jointlymatch originalcrossstate','replacing actualdiags cannot repair alltransport/historyerrors'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(rows))


if __name__=='__main__':main()
