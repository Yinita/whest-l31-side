"""Official finite-bake noise calibration versus exact first Gaussian layer."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--version",required=True)
    args=parser.parse_args()
    root=Path(args.root)
    output=root/f"reference_noise_{args.version}.json"
    assert not output.exists()
    helper_path=Path("experiments/phase2/summarize_covariance_causal_r01.py")
    spec=importlib.util.spec_from_file_location("noise_summary",helper_path)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    journal=(root/"journal.jsonl").read_bytes()
    output.with_suffix(".journal_input.jsonl").write_bytes(journal)
    records=[json.loads(x) for x in journal.decode().splitlines()]
    successful=[x for x in records if x["complete"]]
    stats=[helper.combine([x for x in successful if x["stream"]==stream]) for stream in (0,1)]
    data=Dataset.from_file("benchmark/data/mini_first8/prepared/mini/data.arrow").select_columns(["all_layer_means","avg_variance","sampling_budget_breakdown"])[0]
    bake=json.loads(data["sampling_budget_breakdown"])["provenance"]
    count=bake["chunk_size"]*bake["n_chunks"]
    weights_path=Path("benchmark/data/rpc_inputs/mlp_0.npz")
    weights=np.load(weights_path)["weights"].astype(np.float64)
    variance=np.sum(weights[0]**2,axis=0)
    analytic=np.sqrt(variance)/np.sqrt(2*np.pi)
    true_first_var=variance*(.5-1/(2*np.pi))
    truth=np.asarray(data["all_layer_means"],dtype=np.float64)
    cache_path=Path("results/phase2/covariance_causal_parent_trace_r02.npz")
    cache=np.load(cache_path)
    error=truth-cache["predictions"]
    # Exact first derivative gates: each first-layerpre mean0, Phi=.5.
    first_difference=truth[0]-analytic
    first_forward=[]
    current=first_difference
    for layer in range(16):
        if layer>0:
            current=cache["pre_features"][layer,:,4]*(weights[layer].T@current)
        first_forward.append(current.copy())
    rows=[]
    for layer in range(16):
        nominal=[float(stream["variance"][1,layer].mean()/count) for stream in stats]
        rows.append({"layer":layer,"MC_postvariance_mean_stream":[float(stream["variance"][1,layer].mean()) for stream in stats],
            "official_bake_nominalmean_noise_MSE_stream":nominal,
            "parent_error_mse":float(np.mean(error[layer]**2)),
            "propagated_observed_first_reference_error_mse":float(np.mean(first_forward[layer]**2)),
            "first_error_final_cross_with_parent_error":float(np.mean(first_forward[layer]*error[layer]))})
    report={"rows":rows,"official_bake_sample_count":count,"bake_provenance":bake,
        "first_layer_exact_Gaussian_reference_error_mse":float(np.mean(first_difference**2)),
        "first_layer_exact_postvariance_mean":float(true_first_var.mean()),
        "first_layer_nominal_reference_noise_mse":float(true_first_var.mean()/count),
        "first_layer_parent_vs_exact_analytic_mse":float(np.mean((cache["predictions"][0]-analytic)**2)),
        "first_layer_reference_error_signed_mean":float(first_difference.mean()),
        "first_layer_reference_error_vs_analytic_correlation":float(np.corrcoef(first_difference,analytic)[0,1]),
        "final_official_nominalnoise_MSE":data["avg_variance"]/count,
        "successful_shards":len(successful),"failed_shards":sum(not x["complete"] for x in records),
        "cache_sha256":hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "weights_sha256":hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256":hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        "journal_sha256":hashlib.sha256(journal).hexdigest(),"command":sys.argv,
        "scope":"First layer analyticGaussian exact apartweightsfloat64; bakedreference samplingfinite expectednoise vsobserved onevector not proof. Layernominalnoise usesMCvariancebut officialsameinputcoupled layers, propagatedfirstreferenceerror notindependent oflaterlabels, cannot subtract as estimatedalgorithmerror. BakeGPUdtypeunknown. No candidate or score replacement."}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"firstanalyticMSE":report["first_layer_exact_Gaussian_reference_error_mse"],"firstnominal":report["first_layer_nominal_reference_noise_mse"],"last":rows[-1]},indent=2))


if __name__=="__main__":
    main()
