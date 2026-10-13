"""Fixed-coefficient oracle innovation propagated through the full mean chain."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset


def propagate(effect,cdf,weights):
    current=np.zeros(1024,dtype=np.float64)
    rows=[]
    for layer in range(16):
        current=(np.zeros(1024) if layer==0 else cdf[layer]*(weights[layer].T@current))+effect[layer]
        rows.append(current.copy())
    return np.stack(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--version",required=True)
    parser.add_argument("--chunks",type=int,default=8)
    args=parser.parse_args()
    root=Path(args.root)
    output=root/f"fullchain_{args.version}.json"
    assert not output.exists()
    helpers={}
    for name,path in (("summary",Path("experiments/phase2/summarize_covariance_causal_r01.py")),
                      ("highorder",Path("experiments/phase2/covariance_causal_highorder_r01.py"))):
        spec=importlib.util.spec_from_file_location(f"fullchain_{name}",path)
        helper=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helpers[name]=(helper,path)
    records=[json.loads(x) for x in (root/"journal.jsonl").read_text().splitlines()]
    successful=[x for x in records if x["complete"]]
    cache_path=Path("results/phase2/covariance_causal_parent_trace_r02.npz")
    cache=np.load(cache_path)
    feature=cache["pre_features"]
    weights=np.load("benchmark/data/rpc_inputs/mlp_0.npz")["weights"].astype(np.float64)
    truth=np.asarray(Dataset.from_file("benchmark/data/mini_first8/prepared/mini/data.arrow").select_columns(["all_layer_means"])[0]["all_layer_means"])
    error=truth-cache["predictions"]
    groups={}
    whole={}
    for stream in (0,1):
        selected=sorted([x for x in successful if x["stream"]==stream],key=lambda x:x["shard"])
        assert len(selected)>=args.chunks
        stats=helpers["summary"][0].combine(selected)
        whole[stream]={mode:propagate(value,feature[:,:,4],weights)
                       for mode,value in helpers["highorder"][0].effects(feature,stats).items()}
        groups[stream]=[]
        for subset in np.array_split(selected,args.chunks):
            stats=helpers["summary"][0].combine(list(subset))
            groups[stream].append({mode:propagate(value,feature[:,:,4],weights)
                for mode,value in helpers["highorder"][0].effects(feature,stats).items()})
    rows=[]
    for layer in range(16):
        row={"layer":layer,"parent_error_mse":float(np.mean(error[layer]**2)),"models":{}}
        for mode in ("K2","K3","K4","K2K3","K2K3K4"):
            paired=np.array([np.mean((error[layer]-groups[0][group][mode][layer])*(error[layer]-groups[1][group][mode][layer])) for group in range(args.chunks)])
            residuals=[error[layer]-whole[stream][mode][layer] for stream in (0,1)]
            row["models"][mode]={"chunk_cross_residual_estimates":paired.tolist(),
                "chunk_mean_cross_residual":float(np.mean(paired)),
                "chunk_standard_error":float(np.std(paired,ddof=1)/np.sqrt(args.chunks)),
                "whole_stream_positive_residual_mse":[float(np.mean(value**2)) for value in residuals],
                "whole_average_positive_residual_mse":float(np.mean(((residuals[0]+residuals[1])/2)**2)),
                "whole_stream_cross_residual":float(np.mean(residuals[0]*residuals[1])),
                "whole_residual_stream_difference_mse":float(np.mean((residuals[0]-residuals[1])**2))}
        rows.append(row)
    report={"rows":rows,"successful_shards":len(successful),"failed_shards":sum(not x["complete"] for x in records),
        "samples_per_stream":{str(stream):sum(x["samples"] for x in successful if x["stream"]==stream) for stream in (0,1)},
        "cache_sha256":hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256":{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,(helper,path) in helpers.items()},
        "manifest_sha256":hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
        "journal_sha256":hashlib.sha256((root/"journal.jsonl").read_bytes()).hexdigest(),
        "command":sys.argv,"paired_chunk_count":args.chunks,
        "scope":"Offline firstorderoracle FULLCHAIN innovation effects: fixedtheoryK2/K3/K4derivative, recursively Phi*W.T transport; pretruthmomenterrors themselves includemeanpropagation effects; actualMC inaccessibletocandidate. Entire neutronvector andsameRNG chunk correlations preserved. PositiveMCresidual vs noisecancel bothsaved; earlyGaussian calibration/finitechunks/referencefloor/tracefloats arelimits. Not adjustedscore or deployablealgorithm or tenfoldacceptance."}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"first":rows[0],"last":rows[-1]},indent=2))


if __name__=="__main__":
    main()
