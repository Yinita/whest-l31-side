"""R56 normalized support/component contract with actual features and test heads.

Synthetic coefficients are only negative-control fixtures, no fitted model,
labels/reference/prediction access, candidate directory or score claim.
"""

# ruff: noqa: E402
import os

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
import ast
import copy
import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from build_onpolicy_closure_template_r56 import IDS, candidate_source, validate_model
from tenfold_local_features_r48 import FEATURE_ORDER

OUT = Path("results/phase2/onpolicy_support_contract_r56_r01.json")


def main():
    assert not OUT.exists()
    rng = np.random.default_rng(560013)
    layers = {}
    for layer in range(1, 15):
        mean, scale = rng.normal(size=26), np.exp(rng.normal(size=26))
        mean[0], scale[0] = 0, 1
        layers[str(layer)] = dict(feature_mean=mean.tolist(), feature_scale=scale.tolist(),
                                 theta=rng.normal(scale=1e-7, size=(26, 4)).tolist())
    fixture = dict(feature_order=list(FEATURE_ORDER), n_features=26, train_indices=IDS,
        fitted_layers=list(range(1, 15)), first_layer_zero=True, last_layer_zero=True,
        ridge_lambda=.01, sigma_output_powers=[1, 2, 3, 4], floors=dict(mean=0, variance=1e-10),
        layer_models=layers, training_policy="sequential_on_policy_frozen_previous_heads",
        dataset_revision="cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8",
        h18_sha256="c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2",
        head_capture_provenance={str(i): dict(train_indices=IDS, target_layer=i,
            prefix_layers=list(range(1, i)), capture_manifest_sha256="0"*64,
            preceding_prefix_sha256="0"*64) for i in range(1, 15)})
    support = validate_model(fixture)
    negative = {}
    for name, change in (
        ("missing_layer14", lambda x: x["layer_models"].pop("14")),
        ("heldnetwork_intrain", lambda x: x["train_indices"].append(640)),
        ("wrong_floor", lambda x: x["floors"].update(variance=1e-8)),
        ("wrong_feature_order", lambda x: x["feature_order"].reverse()),
        ("missing_onpolicy_provenance", lambda x: x.pop("head_capture_provenance")),
        ("nonpositive_scale", lambda x: x["layer_models"]["2"]["feature_scale"].__setitem__(3, 0)),
    ):
        altered = copy.deepcopy(fixture)
        change(altered)
        try:
            validate_model(altered)
            negative[name] = False
        except (AssertionError, KeyError):
            negative[name] = True
    assert all(negative.values())
    source = candidate_source("0"*64)
    tree = ast.parse(source)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_corpus_normalized_delta")
    standalone = ast.Module(body=[helper], type_ignores=[])
    namespace = {"fnp": fnp}
    exec(compile(ast.fix_missing_locations(standalone), "r56-isolated-component", "exec"), namespace)
    helper_function = namespace["_corpus_normalized_delta"]
    rows = []
    paths = [Path("results/phase2/modal_joint_oracle_reset_r46/oracle_idx0000.npz"),
             Path("results/phase2/modal_joint_oracle_reset_r46/oracle_idx0016.npz")]
    for path in paths:
        with np.load(path, allow_pickle=False) as bundle:
            # Restrict reads to target-free feature/state/scales snapshots.
            pre = bundle["features"].astype(np.float64)
            state = bundle["states"].astype(np.float64)
            scales = bundle["scales"].astype(np.float64)
        features = np.concatenate([pre, state/scales], axis=-1)
        errors = []
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            legal_support = tuple(fnp.asarray(support[k]) for k in ("feature_mean", "feature_scale", "theta"))
            for layer in range(1, 15):
                head = fixture["layer_models"][str(layer)]
                expected = ((features[layer]-head["feature_mean"])/head["feature_scale"]) @ np.asarray(head["theta"])
                value = helper_function(fnp.asarray(features[layer]), legal_support, layer)
                errors.append(float(np.max(abs(np.asarray(value)-expected))))
        assert max(errors) < 1e-12
        rows.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         normalized_head_numpy_vs_fnp_maxerror=max(errors),
                         component_cost=context.summary_dict()))
    report = dict(version="r56-normalized-onpolicy-support-component-contract", complete=True,
        builder_sha256=hashlib.sha256(Path("experiments/phase2/build_onpolicy_closure_template_r56.py").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows,
        negative_schema_controls=negative, first0_support_theta_zero=bool(np.all(support["theta"][0] == 0)),
        generated_source_no_numpy_torch_import="import numpy" not in source and "import torch" not in source,
        no_actual_model_or_estimator_created=True, labels_reference_prediction_not_read=True,
        fixture_coefficients_only=True,
        scope="supportschema and normalizedhead fixture parity on actualsavedfeatures only; no actualtrainedmodel/wholepredict/strict/score")
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
