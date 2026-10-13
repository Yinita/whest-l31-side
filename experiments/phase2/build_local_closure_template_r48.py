"""Build local26 lawful candidate only from actual frozen per-layer model.

Copies licensed H18-derived R42 body with source provenance. No fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from tenfold_local_features_r48 import FEATURE_ORDER

SOURCE=Path('estimators/tenfold_corpus_per_layer_r42/estimator.py')
OUT=Path('estimators/tenfold_local_closure_r48')


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);a=p.parse_args()
    model=json.loads(a.model.read_text());modelsha=hashlib.sha256(a.model.read_bytes()).hexdigest()
    assert model['feature_order']==list(FEATURE_ORDER) and model['n_features']==26
    assert model['first_layer_zero'] and model['last_layer_zero'] and model['sigma_output_powers']==[1,2,3,4]
    assert model['floors']==dict(mean=0,variance=1e-10)
    assert model['train_indices']==list(range(0,384,16))
    assert set(model['layer_models'])=={str(i) for i in range(1,15)}
    folded=[]
    for layer in range(15):
        if layer==0:folded.append(np.zeros((26,4),dtype=np.float64));continue
        m=model['layer_models'][str(layer)]
        theta=np.asarray(m['theta'],dtype=np.float64);mean=np.asarray(m['feature_mean'],dtype=np.float64);scale=np.asarray(m['feature_scale'],dtype=np.float64)
        assert theta.shape==(26,4) and mean.shape==scale.shape==(26,) and np.min(scale)>0
        assert mean[0]==0 and scale[0]==1
        f=theta/scale[:,None];f[0]-=mean@f;folded.append(f)
    coefficients=np.stack(folded);assert np.isfinite(coefficients).all()
    assert not OUT.exists();OUT.mkdir()
    source=SOURCE.read_text();sourcehash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    source=source.replace("_CORPUS_MODEL_SHA='06187606cb8ceb65e42f76e18f1c510a3a3045af8cd61dff8c424864a00a211f'",f"_CORPUS_MODEL_SHA='{modelsha}'")
    oldorder="_CORPUS_FEATURE_ORDER=('ones','mu','var','sigma','alpha','abs_alpha','phi','Phi','pk1','D3','D21norm','K4_sigma','sigma_phi','skew_H2phi','kurt_H3phi','skew2_H5phi','skewkurt_H6phi','kurt2_H7phi','skew','kurt','gate','depth_fraction')"
    assert source.count(oldorder)==1;source=source.replace(oldorder,"_CORPUS_FEATURE_ORDER="+repr(tuple(FEATURE_ORDER)))
    source=source.replace('support.shape!=(15,22,4)','support.shape!=(15,26,4)')
    source=source.replace('corpus_delta=_corpus_normalized_delta(corpus_f,self._corpus_support,li)','corpus_delta=True  # ownpost features are captured only after ordinary centering')
    start=Path('experiments/phase2/tenfold_local_features_r48.py').read_text().index('def ownpost_features(')
    helper=Path('experiments/phase2/tenfold_local_features_r48.py').read_text()[start:].replace('def ownpost_features(','def _corpus_ownpost_features(')
    helper+= "\n\ndef _corpus_updated_diagonals(k1,k2,k3,k4,sigma,normalized_delta,inherited_delta=None):\n    sigma64=sigma.astype(fnp.float64)\n    change=normalized_delta*fnp.stack([sigma64**p for p in range(1,5)],axis=1)\n    mean=k1.astype(fnp.float64)\n    if inherited_delta is not None:mean=mean+inherited_delta.astype(fnp.float64)\n    mean=fnp.maximum(mean+change[:,0],0.0)\n    if inherited_delta is not None:mean=mean-inherited_delta.astype(fnp.float64)\n    return (mean.astype(fnp.float32),fnp.maximum(k2.astype(fnp.float64)+change[:,1],1e-10).astype(fnp.float32),(k3.astype(fnp.float64)+change[:,2]).astype(fnp.float32),(k4.astype(fnp.float64)+change[:,3]).astype(fnp.float32))\n"
    source=source.replace('\n\nclass Estimator', '\n\n'+helper+'\n\nclass Estimator',1)
    old='''                # Match offline float64 updates followed by copy into existing
                # live float32 cumulants. Existing rider is applied once.
                corpus_sigma64=sigma.astype(fnp.float64)
                corpus_scales=fnp.stack([corpus_sigma64**p for p in range(1,5)],axis=1)
                corpus_change=corpus_delta*corpus_scales
                corpus_base_mean=K1v.astype(fnp.float64)
                if delta is not None:corpus_base_mean=corpus_base_mean+delta.astype(fnp.float64)
                corpus_updated_mean=fnp.maximum(corpus_base_mean+corpus_change[:,0],0.0)
                if delta is not None:corpus_updated_mean=corpus_updated_mean-delta.astype(fnp.float64)
                fnp.copyto(K1v,corpus_updated_mean.astype(fnp.float32))
                fnp.copyto(K2v,fnp.maximum(K2v.astype(fnp.float64)+corpus_change[:,1],1e-10).astype(fnp.float32))
                fnp.copyto(K3v,(K3v.astype(fnp.float64)+corpus_change[:,2]).astype(fnp.float32))
                fnp.copyto(K4v,(K4v.astype(fnp.float64)+corpus_change[:,3]).astype(fnp.float32))'''
    new='''                # Own post state is captured before this layer's intervention.
                corpus_local_f=_corpus_ownpost_features(corpus_f,sigma,K1v,K2v,K3v,K4v,delta)
                corpus_delta=_corpus_normalized_delta(corpus_local_f,self._corpus_support,li)
                corpus_updated=_corpus_updated_diagonals(K1v,K2v,K3v,K4v,sigma,corpus_delta,delta)
                for corpus_original,corpus_value in zip((K1v,K2v,K3v,K4v),corpus_updated):
                    fnp.copyto(corpus_original,corpus_value)'''
    assert source.count(old)==1;source=source.replace(old,new)
    assert 'import numpy' not in source
    (OUT/'estimator.py').write_text(source)
    for name in ('LICENSE','NOTICE.md'):shutil.copy2(SOURCE.parent/name,OUT/name)
    support=OUT/'coefficients.npz';np.savez(support,folded_coefficients=coefficients)
    metadata=dict(complete=True,model=str(a.model),model_sha256=modelsha,support_sha256=hashlib.sha256(support.read_bytes()).hexdigest(),feature_order=list(FEATURE_ORDER),train_layers=list(range(15)),runtime_layers=list(range(1,15)),offline_all_finite=True,dtype='float64',shape=[15,26,4],fit=False,truth_access=False,private_pending_dataset_license=True,builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    support.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    provenance=dict(version='tenfold_local_closure_r48',parent_sha256='c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2',source_body_sha256=sourcehash,candidate_sha256=hashlib.sha256((OUT/'estimator.py').read_bytes()).hexdigest(),model_sha256=modelsha,support_sha256=metadata['support_sha256'],scope='experimental formal diagonal localclosure; actualmodel no scoreclaim',benchmark_accepted=False,wholecandidate_parity=False,strict_resource_acceptance=False,support_private_pending_dataset_license=True)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (OUT/'SOURCE.md').write_text('# R48 lawful local26 frozen closure experimental candidate\n\nUses frozen per-layer linear model with26features: original pre22 float64 features plus ownpost mean(inherited rider once)/σ, K2/σ², K3/σ³, K4/σ⁴. Ownpost captured after original ordinary centering and before any current-layer intervention. All arrays/normalization/features/predict use permitted flopscope operators; support loads with fnp.load and stdlibJSON/hash only. No NumPy import, fitting or labels in candidate.\n\nRuntime completely skips analytic0/final15, intervenes1..14; mean floor0, variance floor1e-10, changeddiags copiedfloat32. Original centered mixedK21/K22 and original pk1/e_b birth ledger retained; Cdiag/S3c/G rebuilt downstream. Arbitraryfour correcteddiags with originalcross are formalclosure, not a proved jointdistribution. Real coefficients remain private until dataset redistribution permission. Component, actualmodelhelper parity, whole-rollout and strictresource gates remain necessary; no tenfold claim.\n')
    exclude=Path(subprocess.check_output(['git','rev-parse','--git-path','info/exclude'],text=True).strip());s=exclude.read_text()
    for suffix in ('npz','json'):
        pattern=f'/estimators/tenfold_local_closure_r48/coefficients.{suffix}'
        if pattern not in s.splitlines():s+='\n'+pattern+'\n'
    exclude.write_text(s)
    print(json.dumps(provenance))


if __name__=='__main__':main()
