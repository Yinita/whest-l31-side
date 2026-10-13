"""Freeze real trained R38 model into two candidate support bundles.

No training/labels, no invented coefficients. Explicit target paths and fresh
artifact policy. Setup-only normalization folding, recorded provenance.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from experiments.phase2.tenfold_mean_shift_r39 import FEATURE_ORDER


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert a.model.is_file() and a.manifest.is_file();assert not a.output.exists()
 model=json.loads(a.model.read_text());manifest=json.loads(a.manifest.read_text());assert manifest['feature_order']==FEATURE_ORDER;assert manifest['train_layers']==list(range(15))
 theta=np.asarray(model['theta'],dtype=float);mean=np.asarray(model['feature_mean'],dtype=float);scale=np.asarray(model['feature_scale'],dtype=float)
 assert theta.shape==(22,4) and mean.shape==scale.shape==(22,);assert np.isfinite(theta).all() and np.isfinite(mean).all() and np.isfinite(scale).all() and np.min(scale)>0
 assert mean[0]==0 and scale[0]==1
 folded=theta/scale[:,None];folded[0]-=mean@folded
 modelsha=hashlib.sha256(a.model.read_bytes()).hexdigest();manifestsha=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
 np.savez(a.output,folded_coefficients=folded.astype(np.float32))
 r={'complete':True,'model':str(a.model),'model_sha256':modelsha,'manifest':str(a.manifest),'manifest_sha256':manifestsha,'support':str(a.output),'support_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'feature_order':FEATURE_ORDER,'train_layers':list(range(15)),'offline_all_finite':True,'dtype':'float32 frozenfolded coefficients','fit':False,'truth_access':False,'no_scoreclaim':True}
 a.output.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
