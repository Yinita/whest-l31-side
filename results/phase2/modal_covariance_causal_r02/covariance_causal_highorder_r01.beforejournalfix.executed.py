"""Offline fixed-coefficient marginal K2/K3/K4 diagnosis with paired input chunks."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset


def effects(feature, stats):
    _, var, sigma, phi, _, _, d3, g4 = [feature[:, :, i] for i in range(8)]
    alpha = feature[:, :, 0]/sigma
    count = int(stats["count"])
    population_var = stats["variance"][0]*(count-1)/count
    true3 = stats["central_third"][0]
    true4 = stats["central_fourth"][0]-3*population_var**2
    delta2 = (stats["variance"][0]-var)*phi/(2*sigma)
    delta3 = (true3-d3)*(-alpha*phi)/(6*sigma**2)
    delta4 = (true4-g4)*((alpha*alpha-1)*phi)/(24*sigma**3)
    return {"K2":delta2, "K3":delta3, "K4":delta4,
            "K2K3":delta2+delta3, "K2K3K4":delta2+delta3+delta4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--chunks", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.root)
    output = root/f"highorder_{args.version}.json"
    assert not output.exists()
    summary_path = Path("experiments/phase2/summarize_covariance_causal_r01.py")
    spec = importlib.util.spec_from_file_location("highorder_summary", summary_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    records = [json.loads(x) for x in (root/"journal.jsonl").read_text().splitlines()]
    successful = [r for r in records if r["complete"]]
    cache_path = Path("results/phase2/covariance_causal_parent_trace_r02.npz")
    cache = np.load(cache_path)
    feature = cache["pre_features"]
    truth = np.asarray(Dataset.from_file("benchmark/data/mini_first8/prepared/mini/data.arrow").select_columns(["all_layer_means"])[0]["all_layer_means"])
    error = truth-cache["predictions"]
    weights = np.load("benchmark/data/rpc_inputs/mlp_0.npz")["weights"].astype(np.float64)
    local = np.stack([error[layer] if layer == 0 else error[layer]-feature[layer,:,4]*(weights[layer].T@error[layer-1]) for layer in range(16)])
    chunks = []
    for stream in (0,1):
        data = sorted([r for r in successful if r["stream"] == stream], key=lambda r:r["shard"])
        assert len(data) >= args.chunks
        for group, subset in enumerate(np.array_split(data,args.chunks)):
            stats = helper.combine(list(subset))
            chunks.append(dict(stream=stream,group=group,count=int(stats["count"]),effect=effects(feature,stats)))
    whole_stats=[helper.combine([r for r in successful if r["stream"]==stream]) for stream in (0,1)]
    whole_effect=[effects(feature,stats) for stats in whole_stats]
    rows=[]
    for layer in range(16):
        row={"layer":layer,"local_mse":float(np.mean(local[layer]**2)),"models":{}}
        for mode in ("K2","K3","K4","K2K3","K2K3K4"):
            residuals, effectcrosses, localcrosses = [], [], []
            for group in range(args.chunks):
                left = next(c for c in chunks if c["stream"]==0 and c["group"]==group)["effect"][mode][layer]
                right = next(c for c in chunks if c["stream"]==1 and c["group"]==group)["effect"][mode][layer]
                effectcross=float(np.mean(left*right))
                localcross=float(np.mean(local[layer]*(left+right)/2))
                residuals.append(row["local_mse"]-2*localcross+effectcross)
                effectcrosses.append(effectcross)
                localcrosses.append(localcross)
            row["models"][mode]={"chunk_pair_residuals":residuals,
                "mean_residual":float(np.mean(residuals)),
                "between_chunk_standard_error":float(np.std(residuals,ddof=1)/np.sqrt(args.chunks)),
                "mean_effect_cross":float(np.mean(effectcrosses)),
                "mean_local_cross":float(np.mean(localcrosses)),
                "whole_stream_residual_mse":[float(np.mean((local[layer]-value[mode][layer])**2)) for value in whole_effect],
                "whole_average_effect_residual_mse":float(np.mean((local[layer]-(whole_effect[0][mode][layer]+whole_effect[1][mode][layer])/2)**2)),
                "whole_independent_stream_difference_mse":float(np.mean((whole_effect[0][mode][layer]-whole_effect[1][mode][layer])**2)),
                "whole_cross_residual_estimate":float(np.mean((local[layer]-whole_effect[0][mode][layer])*(local[layer]-whole_effect[1][mode][layer])))}
        rows.append(row)
    report=dict(rows=rows,version=args.version,successful_shards=len(successful),
        failed_shards=sum(not r["complete"] for r in records),
        chunk_counts=[{k:v for k,v in c.items() if k!="effect"} for c in chunks],
        cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
        journal_sha256=hashlib.sha256((root/"journal.jsonl").read_bytes()).hexdigest(),
        command=sys.argv,paired_chunk_count=args.chunks,
        formula="K2 phi/(2sigma); K3 -alpha*phi/(6sigma²); K4 (alpha²-1)*phi/(24sigma³)",
        scope="Offline oracle marginal-cumulant first-order readout with fixed coefficients, notruthfittedcandidate; interactions/HigherHermite/K3K3/mean-var basis shifts neglected. Entire neuronvector preserves correlation. Limited pairedchunks/success-selection/MCfourthnoise limitations, earlyGaussian calibration mandatory; no adjustedscore or tenfold acceptance")
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"first":rows[0],"last":rows[-1]},indent=2))


if __name__ == "__main__":
    main()
