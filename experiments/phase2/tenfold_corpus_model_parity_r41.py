"""Actual frozen model/helpers lawful arithmetic contract, no labels/reference.

Saved baseline features only plus synthetic state tests. Not whole rollout.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ref_features(inputs,scalar,layer):
    mu,var,sigma,alpha,phi,cdf,pk1,d3,d21norm,g4=[x.astype(np.float64) for x in inputs]
    sk,ku=d3/sigma**3,g4/sigma**4
    hs=[np.ones(len(mu)),alpha]
    for degree in range(1,7):hs.append(alpha*hs[-1]-degree*hs[-2])
    return np.stack([hs[0],mu,var,sigma,alpha,abs(alpha),phi,cdf,pk1,d3,d21norm,np.full(len(mu),scalar),sigma*phi,sk*hs[2]*phi,ku*hs[3]*phi,sk**2*hs[5]*phi,sk*ku*hs[6]*phi,ku**2*hs[7]*phi,sk,ku,cdf,np.full(len(mu),layer/15)],axis=1)


def predict_reference(features,model,mode,layer):
    m=model if mode=='quadratic' else model['layer_models'][layer]
    x=(features-np.asarray(m['feature_mean']))/np.asarray(m['feature_scale'])
    if mode=='quadratic':
        pairs=model['quadratic_pairs']
        expanded=np.concatenate([x,np.stack([x[:,i]*x[:,j] for i,j in pairs],axis=1)],axis=1)
        x=(expanded-np.asarray(m['mapped_mean']))/np.asarray(m['mapped_scale'])
    return x@np.asarray(m['theta'])


def run(mode,name,modelpath):
    path=Path('estimators')/name/'estimator.py'
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=json.loads(modelpath.read_text())
    with flops.BudgetContext(flop_budget=2**41,quiet=True):support,supportsha,modelsha=mod._load_corpus_support()
    assert modelsha==sha(modelpath)
    paths=sorted(Path('results/phase2/modal_joint_closure_pilot_r38_r02').glob('baseline_idx*.npz'))[-8:]
    assert len(paths)==8
    matrix_errors=[];absolute_changes=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as bundle:allfeatures=bundle['features'].copy().astype(float);scales=bundle['scales'].copy().astype(float)
        errors=[]
        for layer in range(1,15):
            features=allfeatures[layer];expected=predict_reference(features,model,mode,layer)*scales[layer]
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                actual=mod._corpus_normalized_delta(fnp.asarray(features),support,layer)*fnp.asarray(scales[layer])
            err=float(np.max(abs(np.asarray(actual)-expected)));errors.append(err);absolute_changes.append(float(np.max(abs(expected))))
        matrix_errors.append(max(errors))
    assert max(matrix_errors)<1e-8,matrix_errors
    rng=np.random.default_rng(261014201);helper_rows=[]
    for n in (8,32,1024):
        mu=rng.normal(size=n).astype(np.float32);var=rng.uniform(.1,2,size=n).astype(np.float32)
        sigma=np.sqrt(var);alpha=mu/sigma;phi=(np.exp(-alpha**2/2)/np.sqrt(2*np.pi)).astype(np.float32)
        cdf=np.asarray([.5*(1+math.erf(v/math.sqrt(2))) for v in alpha],dtype=np.float32)
        pk1=(sigma*phi+mu*cdf).astype(np.float32);d3=(rng.normal(size=n)*.002).astype(np.float32)
        d21norm=rng.uniform(0,.01,size=n).astype(np.float64);g4=(rng.normal(size=n)*.003).astype(np.float32)
        inputs=[mu,var,sigma,alpha,phi,cdf,pk1,d3,d21norm,g4];scalar=float(np.float32(.001));layer=6
        expectedfeatures=ref_features(inputs,scalar,layer)
        expected_delta=predict_reference(expectedfeatures,model,mode,layer)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            ff=mod._corpus_features(*[fnp.asarray(v) for v in inputs],scalar,layer)
            dd=mod._corpus_normalized_delta(ff,support,layer)
        ferror=float(np.max(abs(np.asarray(ff)-expectedfeatures)));derror=float(np.max(abs(np.asarray(dd)-expected_delta)))
        assert ferror<2e-14 and derror<1e-8,(ferror,derror)
        # Match the actual four-channel copyto update, including inherited rider.
        base=rng.normal(size=(n,4)).astype(np.float32);base[:,0]=abs(base[:,0]);base[:,1]=abs(base[:,1])
        rider=(rng.normal(size=n)*1e-4).astype(np.float32);scale=np.stack([sigma.astype(float)**p for p in range(1,5)],axis=1)
        expectedpost=base.astype(float);expectedpost[:,0]+=rider
        expectedpost+=expected_delta*scale;expectedpost[:,0]=np.maximum(expectedpost[:,0],0);expectedpost[:,1]=np.maximum(expectedpost[:,1],1e-10)
        expectedk=expectedpost.copy();expectedk[:,0]-=rider;expectedk=expectedk.astype(np.float32)
        with flops.BudgetContext(flop_budget=2**41,quiet=True):
            sigma64=fnp.asarray(sigma).astype(fnp.float64)
            change=dd*fnp.stack([sigma64**p for p in range(1,5)],axis=1)
            vals=[fnp.asarray(base[:,j].copy()) for j in range(4)]
            rr=fnp.asarray(rider).astype(fnp.float64)
            m=fnp.maximum(vals[0].astype(fnp.float64)+rr+change[:,0],0)-rr
            fnp.copyto(vals[0],m.astype(fnp.float32))
            fnp.copyto(vals[1],fnp.maximum(vals[1].astype(fnp.float64)+change[:,1],1e-10).astype(fnp.float32))
            for j in (2,3):fnp.copyto(vals[j],(vals[j].astype(fnp.float64)+change[:,j]).astype(fnp.float32))
        copyerror=float(np.max(abs(np.stack([np.asarray(v) for v in vals],axis=1)-expectedk)))
        assert copyerror<2e-7,copyerror
        helper_rows.append(dict(width=n,features_maxerror=ferror,normalized_delta_maxerror=derror,float32_diagonal_copy_maxerror=copyerror,feature_model_component_cost=ctx.summary_dict()))
    # Predicate rather than zero model output makes first layer and its floors absent.
    assert 'if riders and 0 < li < 15:' in path.read_text()
    assert 'import numpy' not in path.read_text()
    return dict(mode=mode,candidate_sha256=sha(path),model_sha256=modelsha,support_sha256=supportsha,held_featurepaths_sha256={str(p):sha(p) for p in paths},savedfeatures_delta_maxabs_pernetwork=matrix_errors,max_abs_correction=max(absolute_changes),synthetic_helpers=helper_rows,firstlayer_intervention_skipped=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    rows=[run('quadratic','tenfold_corpus_quadratic_r41',Path('results/phase2/joint_closure_quadratic_cv_r41/frozen_train24_quadratic.json')),run('per_layer','tenfold_corpus_per_layer_r42',Path('results/phase2/joint_closure_per_layer_cv_r42/frozen_train24_per_layer.json'))]
    result=dict(complete=True,version='actual-frozen-corpus-model-parity-r41',rows=rows,source_sha256=sha(Path(__file__)),no_labels_or_reference=True,no_benchmarktruth=True,scope='savedbaselinefeatures + synthetichelper/update parity; not wholeclosedrollout or resourceacceptance')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(mode=r['mode'],savedfeatures_maxerror=max(r['savedfeatures_delta_maxabs_pernetwork']),synthetic_maxerror=max(x['normalized_delta_maxerror'] for x in r['synthetic_helpers']),n1024_component_flops=r['synthetic_helpers'][-1]['feature_model_component_cost']['flops_used']) for r in rows]))


if __name__=='__main__':main()
