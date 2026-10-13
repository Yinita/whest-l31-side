"""Independent-shard cross products preserve correlations across output neurons."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--chunks",type=int,default=4)
    args = parser.parse_args()
    root = Path(args.root)
    output = root/f"chunk_cross_{args.version}.json"
    assert not output.exists()
    summary_path = Path("experiments/phase2/summarize_covariance_causal_r01.py")
    spec = importlib.util.spec_from_file_location("chunk_cross_summary", summary_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    records = [json.loads(x) for x in (root/"journal.jsonl").read_text().splitlines()]
    successful = [r for r in records if r["complete"]]
    cache_path = Path("results/phase2/covariance_causal_parent_trace_r02.npz")
    cache = np.load(cache_path)
    feature = cache["pre_features"]
    variance, sigma, phi, cdf = [feature[:, :, i] for i in (1,2,3,4)]
    truth = np.asarray(Dataset.from_file("benchmark/data/mini_first8/prepared/mini/data.arrow").select_columns(["all_layer_means"])[0]["all_layer_means"])
    error = truth-cache["predictions"]
    weights = np.load("benchmark/data/rpc_inputs/mlp_0.npz")["weights"].astype(np.float64)
    local = np.stack([error[layer] if layer == 0 else error[layer]-cdf[layer]*(weights[layer].T@error[layer-1]) for layer in range(16)])
    chunks = []
    for stream in (0,1):
        data = sorted([r for r in successful if r["stream"] == stream], key=lambda r:r["shard"])
        assert len(data) >= args.chunks
        # Disjoint shard groups, each retaining the complete neuron vector.
        # No assumption of independent neurons is used for error bars.
        for group, subset in enumerate(np.array_split(data,args.chunks)):
            combined = helper.combine(list(subset))
            effect = (combined["variance"][0]-variance)*phi/(2*sigma)
            chunks.append(dict(stream=stream,group=group,count=int(combined["count"]),effect=effect,
                               shard_ids=[r["shard"] for r in subset]))
    pairs = []
    for group in range(args.chunks):
        left = next(c for c in chunks if c["stream"]==0 and c["group"]==group)
        right = next(c for c in chunks if c["stream"]==1 and c["group"]==group)
        effectcross=np.mean(left["effect"]*right["effect"],axis=1)
        localcross=np.mean(local*(left["effect"]+right["effect"])/2,axis=1)
        pairs.append(dict(group=group,left_count=left["count"],right_count=right["count"],
                          effect_cross_per_layer=effectcross.tolist(),local_cross_per_layer=localcross.tolist(),
                          residual_per_layer=(np.mean(local*local,axis=1)-2*localcross+effectcross).tolist()))
    rows=[]
    for layer in range(16):
        values=np.array([r["residual_per_layer"][layer] for r in pairs])
        rows.append(dict(layer=layer,local_mse=float(np.mean(local[layer]**2)),
                         chunk_pair_residual_estimates=values.tolist(),
                         equalchunk_mean=float(np.mean(values)),
                         between_chunk_standard_error=float(np.std(values,ddof=1)/np.sqrt(args.chunks)),
                         limitation="Limited pairedchunk estimates; randomeffects and successfulshard selection remain"))
    report=dict(version=args.version,rows=rows,pairs=pairs,
                chunks=[{k:v for k,v in c.items() if k!="effect"} for c in chunks],
                successful_shards=len(successful),failed_shards=sum(not r["complete"] for r in records),
                cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                helper_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
                journal_sha256=hashlib.sha256((root/"journal.jsonl").read_bytes()).hexdigest(),
                command=sys.argv,paired_chunk_count=args.chunks,
                scope="Offline oracle first-order variance effect. Chunks independent across input RNG, neuronvector kept joint; layer0 analyticGaussian should calibrate nearzero; no benchmark or candidate claim")
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"first":rows[0],"last":rows[-1]},indent=2))


if __name__ == "__main__":
    main()
