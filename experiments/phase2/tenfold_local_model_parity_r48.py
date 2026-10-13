"""Actual R48 model and pre-injection ownstate/helper parity, no labels.

Uses saved teacherforced trajectory features/states only and frozen model.
Not a full closed learned rollout or score claim.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    candidate=Path('estimators/tenfold_local_closure_r48/estimator.py')
    model=json.loads(a.model.read_text());spec=importlib.util.spec_from_file_location('r48candidate',candidate);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert model['feature_order']==list(module._CORPUS_FEATURE_ORDER)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):support,supportsha,modelsha=module._load_corpus_support()
    assert modelsha==sha(a.model)
    paths=sorted(Path('results/phase2/modal_joint_oracle_reset_r46').glob('oracle_idx*.npz'))+sorted(Path('results/phase2/modal_joint_oracle_reset_r47').glob('oracle_idx*.npz'))
    assert len(paths)==24
    rows=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as bundle:
            pre=bundle['features'].copy().astype(float);state=bundle['states'].copy().astype(float);scales=bundle['scales'].copy().astype(float)
        # Arrays reflect pre-reset currentstate with priorlayersoracle-reset.
        # No afterstates, labels, references or prediction arrays are accessed.
        assembled=np.concatenate([pre,state/scales],axis=-1)
        maxerrors=[];maxupdates=[];helpererrors=[]
        for layer in range(1,15):
            lm=model['layer_models'][str(layer)]
            normalized=((assembled[layer]-np.asarray(lm['feature_mean']))/np.asarray(lm['feature_scale']))@np.asarray(lm['theta'])
            expected=normalized*scales[layer]
            sigma=pre[layer,:,3].astype(np.float32)
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                delta=module._corpus_normalized_delta(fnp.asarray(assembled[layer]),support,layer)
            maxerrors.append(float(np.max(abs(np.asarray(delta)*scales[layer]-expected))))
            # Synthetic inherited rider reconstructs saved output mean exactly
            # in float64; ownposthelper must not add it twice.
            rider=np.full(1024,1e-4,dtype=float)
            with flops.BudgetContext(flop_budget=2**41,quiet=True):
                vals=[fnp.asarray(state[layer,:,j]) for j in range(4)]
                k1=fnp.asarray(state[layer,:,0]-rider)
                ff=module._corpus_ownpost_features(fnp.asarray(pre[layer]),fnp.asarray(sigma),k1,*vals[1:],fnp.asarray(rider))
            helpererrors.append(float(np.max(abs(np.asarray(ff)-assembled[layer]))))
            # Actual frozen correction and copy update channels; no inventedcoef.
            base=state[layer].copy();rider32=np.full(1024,1e-4,dtype=np.float32)
            k1=(base[:,0]-rider32).astype(np.float32)
            live=[k1]+[base[:,j].astype(np.float32) for j in range(1,4)]
            basepost=np.stack([live[0].astype(float)+rider32]+[v.astype(float) for v in live[1:]],axis=1)
            expectedpost=basepost+expected
            expectedpost[:,0]=np.maximum(expectedpost[:,0],0);expectedpost[:,1]=np.maximum(expectedpost[:,1],1e-10)
            expectedpost[:,0]-=rider32;expectedpost=expectedpost.astype(np.float32)
            with flops.BudgetContext(flop_budget=2**41,quiet=True):
                out=module._corpus_updated_diagonals(*[fnp.asarray(v) for v in live],fnp.asarray(sigma),delta,fnp.asarray(rider32))
            maxupdates.append(float(np.max(abs(np.stack([np.asarray(v) for v in out],axis=1)-expectedpost))))
        assert max(maxerrors)<1e-10,maxerrors
        assert max(helpererrors)<1e-13,helpererrors
        assert max(maxupdates)<5e-7,maxupdates
        rows.append(dict(path=str(path),sha256=sha(path),folded_vs_originalmodel_delta_maxerror=max(maxerrors),savedstate_ownposthelper_maxerror=max(helpererrors),float32_copyupdate_vs_originalmodel_maxerror=max(maxupdates)))
    assert 'if riders and 0 < li < 15:' in candidate.read_text() and 'import numpy' not in candidate.read_text()
    r=dict(complete=True,version='actualfrozen-local26-model-parity-r48',candidate_sha256=sha(candidate),model_sha256=modelsha,support_sha256=supportsha,source_sha256=sha(Path(__file__)),rows=rows,first0_final15_skipped=True,no_label_reference_or_prediction_access=True,no_scoreclaim=True,scope='actualteacherforcedsavedfeature/model/helper parity; no wholelearnedclosedrollout/resourceacceptance')
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(networks=len(rows),foldederror=max(x['folded_vs_originalmodel_delta_maxerror'] for x in rows),helpererror=max(x['savedstate_ownposthelper_maxerror'] for x in rows),copyerror=max(x['float32_copyupdate_vs_originalmodel_maxerror'] for x in rows))))


if __name__=='__main__':main()
