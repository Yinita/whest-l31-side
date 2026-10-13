"""Compare independent variance streams to frozen canonical predictions, offline only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import Dataset


def combine(records):
    count = 0
    mean = np.zeros((2, 16, 1024), dtype=np.float64)
    second, third, fourth = np.zeros_like(mean), np.zeros_like(mean), np.zeros_like(mean)
    shard_variances = []
    for record in records:
        path = Path(record["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        block = np.load(path)
        b = int(block["count"])
        assert b == record["samples"]
        n = count+b
        delta = block["mean"]-mean
        old_second, old_third = second.copy(), third.copy()
        fourth += (block["m4"]+delta**4*count*b*(count**2-count*b+b*b)/n**3
                   +6*delta**2*(count**2*block["m2"]+b*b*old_second)/n**2
                   +4*delta*(count*block["m3"]-b*old_third)/n)
        third += (block["m3"]+delta**3*count*b*(count-b)/n**2
                  +3*delta*(count*block["m2"]-b*old_second)/n)
        second += block["m2"]+delta**2*count*b/n
        mean += delta*b/n
        count = n
        shard_variances.append(block["m2"]/(b-1))
    assert count > 1
    variance = second/(count-1)
    fourth_central = fourth/count
    variance_se = np.sqrt(np.maximum(fourth_central-variance**2, 0)/count)
    shard_se = np.std(np.stack(shard_variances), axis=0, ddof=1)/np.sqrt(len(records)) if len(records)>1 else np.full_like(mean, np.nan)
    return dict(count=np.asarray(count), mean=mean, variance=variance, central_third=third/count,
                variance_se=variance_se, central_fourth=fourth_central,
                shard_variance_se=shard_se)


def attribution(features, predicted, truth, weights, stream):
    mu, var, sigma, phi, cdf = [features[:, :, i].astype(np.float64) for i in range(5)]
    error = truth-predicted
    variance_delta = stream["variance"][0]-var
    derivative = phi/(2*sigma)
    variance_effect = derivative*variance_delta
    effect_se = derivative*stream["variance_se"][0]
    rows, propagated = [], np.zeros(1024)
    for layer in range(16):
        inherited = np.zeros(1024) if layer == 0 else cdf[layer]*(weights[layer].T@error[layer-1])
        local = error[layer]-inherited
        propagated = (np.zeros(1024) if layer == 0 else cdf[layer]*(weights[layer].T@propagated))+variance_effect[layer]
        denom = np.sum(variance_effect[layer]**2)
        coefficient = float(np.sum(local*variance_effect[layer])/denom) if denom>0 else None
        rows.append({"layer": layer, "mean_error_mse": float(np.mean(error[layer]**2)),
            "inherited_error_mse": float(np.mean(inherited**2)), "local_innovation_mse": float(np.mean(local**2)),
            "variance_bias_rms": float(np.sqrt(np.mean(variance_delta[layer]**2))),
            "variance_effect_mse": float(np.mean(variance_effect[layer]**2)),
            "variance_effect_mc_noise_mse": float(np.mean(effect_se[layer]**2)),
            "local_minus_variance_effect_mse": float(np.mean((local-variance_effect[layer])**2)),
            "diagnostic_oracle_best_fit_coefficient": coefficient,
            "variance_effect_local_cross": float(np.mean(local*variance_effect[layer])),
            "propagated_mean_error_after_oracle_variance_mse": float(np.mean((error[layer]-propagated)**2)),
            "pre_mean_sampling_error_vs_pred_rms": float(np.sqrt(np.mean((stream["mean"][0, layer]-mu[layer])**2)))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--root", default="results/phase2/modal_covariance_causal_r01")
    args = parser.parse_args()
    root = Path(args.root)
    output = root/f"attribution_{args.version}.json"
    aggregate_path = root/f"aggregate_{args.version}.npz"
    assert not output.exists() and not aggregate_path.exists()
    manifest = json.loads((root/"manifest.json").read_text())
    journal_bytes = (root/"journal.jsonl").read_bytes()
    journal_snapshot = root/f"attribution_{args.version}.journal_input.jsonl"
    assert not journal_snapshot.exists()
    journal_snapshot.write_bytes(journal_bytes)
    journal = [json.loads(line) for line in journal_bytes.decode().splitlines()]
    successful = [r for r in journal if r["complete"]]
    assert len({(r["stream"],r["shard"]) for r in successful}) == len(successful)
    streams = {s:combine([r for r in successful if r["stream"] == s]) for s in (0,1)}
    np.savez(aggregate_path, **{f"stream{s}_{k}":v for s, data in streams.items() for k,v in data.items()})
    data = Dataset.from_file("benchmark/data/mini_first8/prepared/mini/data.arrow").select_columns(["mlp_name","mlp_seed","weights","all_layer_means"])[0]
    assert data["mlp_name"] == manifest["mlp_name"] and data["mlp_seed"] == 6319981554997072999
    # r01's field mlp_seed mistakenly holds converted grader setup seed;
    # frozen weights/name and independent official raw seed checked here.
    setup_seed = manifest.get("grader_setup_seed", manifest["mlp_seed"])
    assert setup_seed == 3909487726
    assert manifest["mlp_seed"] in (3909487726, 6319981554997072999)
    weights = np.load(manifest["weights"])["weights"].astype(np.float64)
    assert np.array_equal(weights, np.asarray(data["weights"], dtype=np.float32))
    truth = np.asarray(data["all_layer_means"],dtype=np.float64)
    cached = Path("benchmark/data/v29_g4_v30_102_r03/network_0.npz")
    cache = np.load(cached)
    base = cache["base_features"]
    features = base[:, :, [1,2,3,6,7]]
    models = {"V29":(features,cache["predictions"],cached)}
    h18 = Path("results/phase2/covariance_causal_parent_trace_r01.npz")
    if h18.exists():
        hp = np.load(h18)
        models["V37bH18"] = (hp["pre_features"],hp["predictions"],h18)
    rows = {name:{str(s):attribution(f,p,truth,weights,stream) for s,stream in streams.items()}
            for name,(f,p,path) in models.items()}
    diff = streams[0]["variance"]-streams[1]["variance"]
    report = {"version":args.version, "rows":rows, "complete_jobs":len(successful),
        "failures":[r for r in journal if not r["complete"]],
        "samples_per_stream":{str(s):int(d["count"]) for s,d in streams.items()},
        "pre_variance_stream_difference_rms_per_layer":np.sqrt(np.mean(diff[0]**2,axis=1)).tolist(),
        "aggregate":str(aggregate_path), "aggregate_sha256":hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
        "model_caches":{name:{"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for name,(f,p,path) in models.items()},
        "dataset_commit":"aa99830fdc09fad15407b10e8e3459d3e18bba0a", "weights_sha256":manifest["weights_sha256"],
        "official_raw_mlp_seed":data["mlp_seed"], "grader_setup_seed":setup_seed,
        "seed_provenance_correction":"Running r01 manifest mlp_seed mislabeled grader seed; frozen file unchanged, rawseed/name/weights guards used in this report",
        "scope":"Offline first-order derivative phi/(2sigma) variance causal attribution; exact MC variance is an oracle unavailable to candidates; reference and derivative linearization error remain. One public development network, no adjusted or holdout result"}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"samples":report["samples_per_stream"],
        "final_rows":{name:{s:r[-1] for s,r in group.items()} for name,group in rows.items()}},indent=2),flush=True)


if __name__ == "__main__":
    main()
