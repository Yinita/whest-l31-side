"""Fixed V29 mixed orbit four-output direction check, offline diagnosis only."""

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
    output=root/f"r18_direction_{args.version}.json"
    assert not output.exists()
    helper_path=Path("experiments/phase2/summarize_covariance_causal_r01.py")
    spec=importlib.util.spec_from_file_location("r18direction_summary",helper_path)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    journal_bytes=(root/"journal.jsonl").read_bytes()
    output.with_suffix(".journal_input.jsonl").write_bytes(journal_bytes)
    records=[json.loads(x) for x in journal_bytes.decode().splitlines()]
    successful=[x for x in records if x["complete"]]
    stats=[helper.combine([x for x in successful if x["stream"]==stream]) for stream in (0,1)]
    signal_path=Path("results/phase2/tenfold_mixed_fullhistory_readout_gate_r18.json")
    signal=json.loads(signal_path.read_text())
    parent_path=Path("estimators/k3_v29/estimator.py")
    assert signal["parent_sha256"]==hashlib.sha256(parent_path.read_bytes()).hexdigest()
    cache_path=Path("benchmark/data/v29_g4_v30_102_r03/network_0.npz")
    data=np.load(cache_path)
    baseline=data["base_features"][15,:,9]
    rows=[]
    for record in signal["rows"]:
        index=record["output"]
        true=[float(stream["central_third"][0,15,index]) for stream in stats]
        delta=[value-float(baseline[index]) for value in true]
        amount=record["mixed_next_d3_signal"]
        rows.append({"canonical_output":index,"V29_predicted_pre_K3":float(baseline[index]),
            "MC_true_pre_K3_streams":true,"remaining_K3_delta_streams":delta,
            "r18_exact_mixed_orbit_signal":amount,
            "same_direction_streams":[bool(value*amount>0) for value in delta],
            "fixed4output_before_squared_error_streams":[value**2 for value in delta],
            "fixed4output_after_squared_error_streams":[(value-amount)**2 for value in delta]})
    report={"rows":rows,"successful_shards":len(successful),"failed_shards":sum(not x["complete"] for x in records),
        "samples_per_stream":[int(stream["count"]) for stream in stats],
        "journal_sha256":hashlib.sha256(journal_bytes).hexdigest(),
        "manifest_sha256":hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
        "V29_cache_sha256":hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "signal_report_sha256":hashlib.sha256(signal_path.read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"command":sys.argv,
        "scope":"Four fixedcanonical outputs only; V29sameparentSHA no permutation, r18sourcecapture and V29D3cachefromdifferentcalls with numericaldrift; H18notcomparable. MCmoment truth is offlineoracle, fixedmathematicalorbitcoefficient1 no fit; not holdoutscore or candidate accuracy improvement, fulloutput/history stillpending."}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"rows":rows},indent=2))


if __name__=="__main__":
    main()
