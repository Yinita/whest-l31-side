"""Two actual savedcanonical final26 feature/input lawful component parity.

Reads only feature/raw/sigma/ownpost/expectedparentmean arrays, no prediction,
MC labels, reference, model, fitting or cloud work.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from experiments.phase2.tenfold_final_features_r52 import final_features,prefeatures,raw_post_cumulants,FEATURE_ORDER


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    rows=[]
    paths=sorted(Path('results/phase2/final_readout_capture_r52').glob('capture_idx*.npz'))[:2];assert len(paths)==2
    for path in paths:
        with np.load(path,allow_pickle=False) as b:
            captured=b['features'].copy();raw=b['raw_power'].copy();sigma=b['sigma'].copy();own=b['ownpost'].copy();expectedmean=b['expected_final'].copy()
        assert captured.shape==(1024,26) and raw.shape==(4,1024)
        assert np.array_equal(captured[:,10],np.zeros(1024)) and np.array_equal(captured[:,21],np.ones(1024))
        # Current default NO_CORR=1: original actual emittedmean is rawpk1.
        assert np.array_equal(expectedmean,raw[0])
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            pre=fnp.asarray(captured[:,:22]);rr=fnp.asarray(raw.astype(np.float32));ss=fnp.asarray(sigma.astype(np.float32))
            assembled=final_features(pre,ss,rr);actualown=raw_post_cumulants(rr)
            featurespre=prefeatures(*[fnp.asarray(captured[:,j].astype(np.float32)) for j in (1,2,3,4,6,7,8,9)],
                fnp.asarray((captured[:,19]*sigma**4).astype(np.float32)),float(captured[0,11]))
        # g4 reconstructed from saved kurt can introduce multiplicationroundoff;
        # ownpre22exceptg4derivedcolumns should remain original capture.
        error=float(np.max(abs(np.asarray(assembled)-captured)));ownerror=float(np.max(abs(np.asarray(actualown)-own)))
        assert error<1e-10 and ownerror<1e-10,(error,ownerror)
        originalcols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,18,20,21]
        preerror=float(np.max(abs(np.asarray(featurespre)[:,originalcols]-captured[:,originalcols])))
        assert preerror<1e-10,preerror
        # All saved inputs and outputs are canonical; internally permute every
        # vector/rawcolumn consistently, then inverse returns savedcanonical26.
        perm=np.roll(np.arange(1024),1);inverse=np.argsort(perm)
        with flops.BudgetContext(flop_budget=2**41,quiet=True):
            internal=final_features(fnp.asarray(captured[perm,:22]),fnp.asarray(sigma[perm].astype(np.float32)),fnp.asarray(raw[:,perm].astype(np.float32)))
        perr=float(np.max(abs(np.asarray(internal)[inverse]-captured)));assert perr<1e-10
        # Rider computes f32sum before f64; wrong promoting-before-sum is detected.
        rider=np.full(1024,1e-4,dtype=np.float32);expectedridden=own.copy();expectedridden[:,0]=(raw[0].astype(np.float32)+rider).astype(float)
        with flops.BudgetContext(flop_budget=2**41,quiet=True):ridden=raw_post_cumulants(fnp.asarray(raw.astype(np.float32)),fnp.asarray(rider))
        ridererror=float(np.max(abs(np.asarray(ridden)-expectedridden)));assert ridererror<1e-10
        wrongfloat64mean=(raw[0]+rider).astype(float);negative=float(np.max(abs(wrongfloat64mean-expectedridden[:,0])));assert negative>0
        rows.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),saved_canonical_final26_maxerror=error,saved_canonical_ownpost4_maxerror=ownerror,prefeatureoriginalcolumns_maxerror=preerror,canonical_internal_roundtrip_maxerror=perr,inherited_rider_fp32beforepromote_maxerror=ridererror,wrong_early_f64_rider_negative_control_error=negative,legal_component_cost=ctx.summary_dict(),scope='raw/postcanonical parity; reconstructedg4columns excluded from pre22roundtrip'))
    result=dict(complete=True,version='actualsaved-final26-fnpcomponent-r52',feature_order=FEATURE_ORDER,rows=rows,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path('experiments/phase2/tenfold_final_features_r52.py'))},no_labels_reference_prediction_model_access=True,fit=False,cloud_started=False,no_scoreclaim=True,scope='twoactualsavedcanonical finalfeatures/rawinput component; no learnedmodel or wholecandidateparity')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps([dict(path=r['path'],maxerror=r['saved_canonical_final26_maxerror'],flops=r['legal_component_cost']['flops_used']) for r in rows]))


if __name__=='__main__':main()
