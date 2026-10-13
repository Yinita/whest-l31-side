"""Offline pre-to-post marginal-moment production error attribution."""

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
    output=root/f"postproduction_{args.version}.json"
    assert not output.exists()
    helper_path=Path("experiments/phase2/summarize_covariance_causal_r01.py")
    spec=importlib.util.spec_from_file_location("postprod_summary",helper_path)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    journal_bytes=(root/"journal.jsonl").read_bytes()
    journal_snapshot=output.with_suffix(".journal_input.jsonl")
    assert not journal_snapshot.exists()
    journal_snapshot.write_bytes(journal_bytes)
    records=[json.loads(x) for x in journal_bytes.decode().splitlines()]
    successful=[x for x in records if x["complete"]]
    stats=[helper.combine([x for x in successful if x["stream"]==stream]) for stream in (0,1)]
    cache_path=Path("results/phase2/covariance_causal_parent_trace_post_r03.npz")
    feature=np.load(cache_path)["pre_features"].astype(np.float64)
    rows=[]
    for layer in range(16):
        mu,var,sigma,phi,cdf,pk1,d3,g4,pk2,_,_=feature[layer].T
        alpha=mu/sigma
        row={"layer":layer,"streams":[]}
        for stream in stats:
            count=int(stream["count"])
            pv=stream["variance"][0,layer]
            pk3true=stream["central_third"][0,layer]
            pk4true=stream["central_fourth"][0,layer]-3*(pv*(count-1)/count)**2
            delta_mu=stream["mean"][0,layer]-mu
            delta_var=pv-var
            delta3=pk3true-d3
            delta4=pk4true-g4
            mean_linear=(cdf*delta_mu+phi/(2*sigma)*delta_var
                         -alpha*phi/(6*sigma**2)*delta3
                         +(alpha**2-1)*phi/(24*sigma**3)*delta4)
            # f=ReLU²: f'=2ReLU, f''=2H, f'''=2delta,
            # f''''=2delta'; fixed linear cumulant response.
            marginal_gaussian_mean=sigma*phi+mu*cdf
            second_linear=(2*marginal_gaussian_mean*delta_mu+cdf*delta_var
                           +phi/(3*sigma)*delta3-alpha*phi/(12*sigma**2)*delta4)
            true_mean=stream["mean"][1,layer]
            true_second=stream["variance"][1,layer]*(count-1)/count+true_mean**2
            corrected_mean=pk1+mean_linear
            corrected_second=pk2+second_linear
            baseline_variance=pk2-pk1**2
            corrected_variance=corrected_second-corrected_mean**2
            row["streams"].append({"samples":count,
                "postmean_error_mse":float(np.mean((true_mean-pk1)**2)),
                "postsecond_error_mse":float(np.mean((true_second-pk2)**2)),
                "postvariance_error_mse":float(np.mean((stream["variance"][1,layer]-baseline_variance)**2)),
                "postmean_after_oracle_pre_cumulants_mse":float(np.mean((true_mean-corrected_mean)**2)),
                "postsecond_after_oracle_pre_cumulants_mse":float(np.mean((true_second-corrected_second)**2)),
                "postvariance_after_oracle_pre_cumulants_mse":float(np.mean((stream["variance"][1,layer]-corrected_variance)**2)),
                "postsecond_error_signedmean":float(np.mean(true_second-pk2)),
                "postsecond_after_signedmean":float(np.mean(true_second-corrected_second)),
                "p2_localmodel_correction_rms":float(np.sqrt(np.mean(second_linear**2)))})
        rows.append(row)
    report={"rows":rows,"successful_shards":len(successful),"failed_shards":sum(not x["complete"] for x in records),
        "cache_sha256":hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256":hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        "manifest_sha256":hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
        "journal_sha256":hashlib.sha256(journal_bytes).hexdigest(),"command":sys.argv,
        "scope":"Offline sameinput pre/post moment diagnostic; MCerrors correlated beneficially, no neuronIID assumption/holdoutscore. Pre-mu/variance/K3/K4 derivative atparentGaussian notfullnonlinearclosure, no fittedcandidate. Layer0 checks exactGaussian kernel; MCnoise/finitecentralcumulantbias/interactions remain."}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"first":rows[0],"last":rows[-1]},indent=2))


if __name__=="__main__":
    main()
