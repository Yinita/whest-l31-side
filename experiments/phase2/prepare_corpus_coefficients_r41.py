"""Convert actual frozen R41/R42 models to private lawful support artifacts.

No labels, training, fitting or competition references. Fresh-output only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

FEATURES = ['ones','mu','var','sigma','alpha','abs_alpha','phi','Phi','pk1','D3','D21norm','K4_sigma','sigma_phi','skew_H2phi','kurt_H3phi','skew2_H5phi','skewkurt_H6phi','kurt2_H7phi','skew','kurt','gate','depth_fraction']
PAIRS = [(i,j) for i in range(1,22) for j in range(i,22)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--kind',choices=['quadratic','per_layer'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix('.json').exists()
    model=json.loads(args.model.read_text())
    assert model['feature_order']==FEATURES
    assert model['train_indices']==list(range(0,384,16))
    if args.kind=='quadratic':
        assert model['quadratic'] and model['quadratic_pairs']==[list(x) for x in PAIRS]
        shapes=dict(feature_mean=(22,),feature_scale=(22,),mapped_mean=(253,),mapped_scale=(253,),theta=(253,4))
        arrays={name:np.asarray(model[name],dtype=np.float64) for name in shapes}
        assert all(v.shape==shapes[name] for name,v in arrays.items())
        assert np.min(arrays['feature_scale'])>0 and np.min(arrays['mapped_scale'])>0
    else:
        assert len(model['layer_models'])==15
        rows=[]
        for layer in model['layer_models']:
            theta=np.asarray(layer['theta'],dtype=np.float64)
            mean=np.asarray(layer['feature_mean'],dtype=np.float64)
            scale=np.asarray(layer['feature_scale'],dtype=np.float64)
            assert theta.shape==(22,4) and mean.shape==scale.shape==(22,)
            assert mean[0]==0 and scale[0]==1 and np.min(scale)>0
            folded=theta/scale[:,None]
            folded[0]-=mean@folded
            rows.append(folded)
        arrays=dict(folded_coefficients=np.stack(rows))
        assert np.array_equal(arrays['folded_coefficients'][0],np.zeros((22,4)))
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez(args.output,**arrays)
    metadata=dict(complete=True,model=str(args.model),model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),support=str(args.output),support_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),kind=args.kind,feature_order=FEATURES,train_layers=list(range(15)),runtime_layers=list(range(1,15)),offline_all_finite=True,dtype='float64',shapes={k:list(v.shape) for k,v in arrays.items()},fit=False,truth_access=False,no_scoreclaim=True,private_pending_dataset_license=True)
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('feature_order','shapes')}))


if __name__=='__main__':main()
