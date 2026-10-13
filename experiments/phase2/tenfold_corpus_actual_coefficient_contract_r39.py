"""Real frozen coefficient parity using candidate helpers, no labels/reference."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 modelpath=Path('results/phase2/modal_joint_closure_pilot_r38_r02/frozen_joint_ridge.json');m=json.loads(modelpath.read_text());rng=np.random.default_rng(261014170);rows=[]
 # Saved held-network baseline prefeatures, not labels or validationreference.
 paths=sorted(Path('results/phase2/modal_joint_closure_pilot_r38_r02').glob('baseline_idx*.npz'))[-8:]
 assert len(paths)==8
 for mode,name in [('mean','tenfold_corpus_mean_r39'),('joint','tenfold_corpus_joint_r40')]:
  source=Path('estimators')/name/'estimator.py';spec=importlib.util.spec_from_file_location(name,source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:coef,supportsha,modelsha=mod._load_corpus_support(mode)
  assert modelsha==hashlib.sha256(modelpath.read_bytes()).hexdigest();errors=[];maxchange=[]
  for path in paths:
   with np.load(path) as bundle:features=bundle['features'].copy().astype(float);scales=bundle['scales'].copy().astype(float)
   ref=((features-np.asarray(m['feature_mean']))/np.asarray(m['feature_scale']))@np.asarray(m['theta']);ref=ref*scales
   cols=1 if mode=='mean' else 4
   with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    actual=(fnp.asarray(features.astype(np.float32))@coef)*fnp.asarray(scales[...,:cols].astype(np.float32))
   diff=np.asarray(actual).astype(float)-ref[...,:cols];errors.append(float(np.max(abs(diff))));maxchange.append(float(np.max(abs(ref[...,:cols]))))
  # New copiedbody must contain no numpy import, fixedsupportloadonlysetup.
  assert 'import numpy' not in source.read_text()
  rows.append({'mode':mode,'candidate_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'support_sha256':supportsha,'frozen_model_sha256':modelsha,'baseline_featurepaths_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'fp32_folded_delta_maxabs_pernetwork':errors,'f64delta_maxabs_pernetwork':maxchange,'legal_last_delta_component_cost':ctx.summary_dict(),'no_label_access':True})
  assert max(errors)<1e-7,errors
 r={'version':'tenfold-corpus-actual-coefficient-contract-r39','complete':True,'rows':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'actualfrozenr38 coefficients,heldnetwork saved baselinefeatures only; no closedrollout/numericproductionparityclaim','no_benchmarktruth':True,'not_resource_acceptance':True}
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'rows':[{'mode':r['mode'],'maximum_folded_error':max(r['fp32_folded_delta_maxabs_pernetwork'])} for r in rows]}))
if __name__=='__main__':main()
