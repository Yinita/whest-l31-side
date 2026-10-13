"""Late marginal bias and activation-weighted D3, offline fixed diagnostics."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--version",required=True)
    args=parser.parse_args()
    root=Path(args.root)
    output=root/f"signed_bias_{args.version}.json"
    assert not output.exists()
    helper_path=Path("experiments/phase2/summarize_covariance_causal_r01.py")
    spec=importlib.util.spec_from_file_location("signed_bias_summary",helper_path)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    journal_bytes=(root/"journal.jsonl").read_bytes()
    journal_snapshot=output.with_suffix(".journal_input.jsonl")
    assert not journal_snapshot.exists()
    journal_snapshot.write_bytes(journal_bytes)
    records=[json.loads(x) for x in journal_bytes.decode().splitlines()]
    successful=[x for x in records if x["complete"]]
    stats=[helper.combine([x for x in successful if x["stream"]==stream]) for stream in (0,1)]
    cache_path=Path("results/phase2/covariance_causal_parent_trace_r02.npz")
    feature=np.load(cache_path)["pre_features"]
    rows=[]
    for layer in range(8,16):
        row={"layer":layer}
        for name,idx,order in (("K2",1,2),("K3",6,3),("K4",7,4)):
            truth=[]
            for stream in stats:
                if order==2:
                    value=stream["variance"][0,layer]
                elif order==3:
                    value=stream["central_third"][0,layer]
                else:
                    value=stream["central_fourth"][0,layer]-3*(stream["variance"][0,layer]*(int(stream["count"])-1)/int(stream["count"]))**2
                truth.append(value)
            predicted=feature[layer,:,idx]
            scale=feature[layer,:,2]**order
            delta=[(value-predicted)/scale for value in truth]
            normalized=predicted/scale
            row[name]={"parent_normalized_rms":float(np.sqrt(np.mean(normalized**2))),
                "true_normalized_rms_stream":[float(np.sqrt(np.mean((value/scale)**2))) for value in truth],
                "delta_normalized_rms_stream":[float(np.sqrt(np.mean(value**2))) for value in delta],
                "delta_cross_normalized_MSE":float(np.mean(delta[0]*delta[1])),
                "signed_delta_parent_cosine_stream":[float(np.dot(value,normalized)/max(np.linalg.norm(value)*np.linalg.norm(normalized),1e-30)) for value in delta],
                "mean_normalized_delta_stream":[float(np.mean(value)) for value in delta]}
        sigma=feature[layer,:,2]
        alpha=feature[layer,:,0]/sigma
        coefficient=-alpha*feature[layer,:,3]/(6*sigma**2)
        predicted=feature[layer,:,6]
        delta=[stream["central_third"][0,layer]-predicted for stream in stats]
        mask=abs(alpha)<4
        row["D3_activation_weighted"]={"non_saturated_count":int(mask.sum()),
            "active_delta_standardized_rms":[float(np.sqrt(np.mean((value[mask]/sigma[mask]**3)**2))) for value in delta],
            "parent_active_rms":float(np.sqrt(np.mean((predicted[mask]/sigma[mask]**3)**2))),
            "readout_effect_rms":[float(np.sqrt(np.mean((value*coefficient)**2))) for value in delta],
            "zero_parent_count":int((predicted==0).sum()),
            "effect_in_zero_parent_cross":float(np.mean(delta[0]*coefficient*delta[1]*coefficient*(predicted==0))),
            "effect_total_cross":float(np.mean(delta[0]*coefficient*delta[1]*coefficient))}
        # Pure globalscalevariance/radial proxy is an oracle attribution;
        # no fitted correction enters submitted code.
        predicted_var=feature[layer,:,1]
        variance_delta=[stream["variance"][0,layer]-predicted_var for stream in stats]
        global_scale=[float(np.dot(value,predicted_var)/np.dot(predicted_var,predicted_var)) for value in variance_delta]
        residual=[value-coef*predicted_var for value,coef in zip(variance_delta,global_scale)]
        row["variance_global_scale"]={"relative_delta_stream":global_scale,
            "raw_delta_cross":float(np.mean(variance_delta[0]*variance_delta[1])),
            "scale_removed_delta_cross":float(np.mean(residual[0]*residual[1]))}
        rows.append(row)
    report={"rows":rows,"samples_per_stream":[int(stream["count"]) for stream in stats],
        "successful_shards":len(successful),"failed_shards":sum(not x["complete"] for x in records),
        "cache_sha256":hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256":hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        "manifest_sha256":hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
        "journal_sha256":hashlib.sha256(journal_bytes).hexdigest(),
        "command":sys.argv,
        "scope":"Offline signedmoments andactivationweightedD3 diagnostic, parent sigma normalization; MCfinitehighermomentnoise and successfulshardselection retained. Globalvarianceprojection onlyoracle attribution, no fittedcandidate or stageclaim."}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"last":rows[-1]},indent=2))


if __name__=="__main__":
    main()
