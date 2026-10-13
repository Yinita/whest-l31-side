"""Actual frozen normalized R56 support parity on target-free R54 saved states.

No reference/labels/prediction arrays, fitting, wholepredict or score claims.
Read available actual full24captures; archive payload storage may be sparse.
"""

# ruff: noqa: E402
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from build_onpolicy_closure_template_r56 import validate_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and sha(args.model) == args.expected_model_sha256
    model = json.loads(args.model.read_text())
    validate_model(model)
    spec = importlib.util.spec_from_file_location("r56_actualmodel", args.candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        support, supportsha, modelsha = module._load_corpus_support()
    assert modelsha == args.expected_model_sha256
    records = [json.loads(line) for line in (args.experiment / "journal.jsonl").read_text().splitlines()]
    actual = [r for r in records if r.get("complete") and r.get("branch") == "full24"
              and r.get("local_payload_stored") and Path(r["path"]).is_file()]
    assert actual, "No actual target-free full24saved capture for model component parity"
    rows = []
    for row in actual:
        path, layer = Path(row["path"]), row["target_layer"]
        assert sha(path) == row["payload_sha256"]
        with np.load(path, allow_pickle=False) as bundle:
            features = bundle["features26"].astype(np.float64)
            state = bundle["state4"].astype(np.float64)
            scales = bundle["scales4"].astype(np.float64)
            rider = bundle["inherited_delta"].astype(np.float32)
        assert features.shape == (1024, 26) and state.shape == scales.shape == (1024, 4)
        head = model["layer_models"][str(layer)]
        expected = ((features-head["feature_mean"])/head["feature_scale"]) @ np.asarray(head["theta"])
        sigma = features[:, 3].astype(np.float32)
        live = [(state[:, 0]-rider.astype(np.float64)).astype(np.float32)]
        live += [state[:, j].astype(np.float32) for j in range(1, 4)]
        reconstructed = np.stack([live[0].astype(np.float64)+rider.astype(np.float64)]
                                 + [v.astype(np.float64) for v in live[1:]], axis=1)
        assert np.max(abs(reconstructed-state)) < 2e-6
        updated = reconstructed+expected*scales
        updated[:, 0] = np.maximum(updated[:, 0], 0)-rider.astype(np.float64)
        updated[:, 1] = np.maximum(updated[:, 1], 1e-10)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            delta = module._corpus_normalized_delta(fnp.asarray(features), support, layer)
            after = module._corpus_updated_diagonals(*[fnp.asarray(v) for v in live],
                fnp.asarray(sigma), delta, fnp.asarray(rider))
        deltaerror = float(np.max(abs(np.asarray(delta)-expected)))
        updateerror = float(np.max(abs(np.stack([np.asarray(v) for v in after], axis=1)
                                       -updated.astype(np.float32))))
        assert deltaerror < 1e-11 and updateerror < 5e-7
        rows.append(dict(idx=row["idx"], layer=layer, path=str(path), sha256=sha(path),
            normalized_delta_maxerror=deltaerror, copied_diagonal_maxerror=updateerror,
            reconstructed_state_error=float(np.max(abs(reconstructed-state))),
            component_cost=context.summary_dict()))
    report = dict(version="r56-actualfullfrozen-onpolicy-model-component-parity", complete=True,
        candidate_sha256=sha(args.candidate), model_sha256=modelsha, support_sha256=supportsha,
        source_sha256=sha(Path(__file__)), rows=rows,
        covered_layers=sorted({r["layer"] for r in rows}), labels_reference_prediction_not_read=True,
        fitting_performed=False, whole_predict_parity=False, strict_acceptance=False,
        scope="actualfrozenmodel/component parity only; savedfeature snapshots sparse, no whole rollout or score")
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
