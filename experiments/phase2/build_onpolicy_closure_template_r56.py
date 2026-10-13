"""Prepare a lawful R56 audit template; build only from a frozen full14 model.

No fitting. Model/coefficients stay private. No input model means no estimator.
Preserve R54 normalized-head operation order instead of folded affine support.
"""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from tenfold_local_features_r48 import FEATURE_ORDER

SOURCE = Path("estimators/tenfold_local_closure_r48/estimator.py")
SOURCE_SHA = "e177526a1781091f7127388891f935598c30c627d817dacff063612f2faea0e2"
DEFAULT_OUT = Path("estimators/tenfold_onpolicy_closure_r56")
IDS = list(range(0, 384, 16))
OLD_MODEL_SHA = "f50a3ac0990ed659dd39dd66ff18a998a733305a09a8bc4dbcda5fe32bc5f559"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_model(model, require_provenance=True):
    assert model["feature_order"] == list(FEATURE_ORDER) and model["n_features"] == 26
    assert model["train_indices"] == IDS and model["fitted_layers"] == list(range(1, 15))
    assert model["first_layer_zero"] and model["last_layer_zero"]
    assert model["ridge_lambda"] == .01 and model["sigma_output_powers"] == [1, 2, 3, 4]
    assert model["floors"] == dict(mean=0, variance=1e-10)
    assert set(model["layer_models"]) == {str(i) for i in range(1, 15)}
    if require_provenance:
        assert model["training_policy"] == "sequential_on_policy_frozen_previous_heads"
        assert model["dataset_revision"] == "cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8"
        assert model["h18_sha256"] == "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
        assert set(model["head_capture_provenance"]) == {str(i) for i in range(1, 15)}
        for layer in range(1, 15):
            record = model["head_capture_provenance"][str(layer)]
            assert record["train_indices"] == IDS and record["target_layer"] == layer
            assert record["prefix_layers"] == list(range(1, layer))
            assert len(record["capture_manifest_sha256"]) == 64
            assert len(record["preceding_prefix_sha256"]) == 64
    mean = np.zeros((15, 26), dtype=np.float64)
    scale = np.ones((15, 26), dtype=np.float64)
    theta = np.zeros((15, 26, 4), dtype=np.float64)
    for layer in range(1, 15):
        head = model["layer_models"][str(layer)]
        m, s, t = [np.asarray(head[k], dtype=np.float64)
                   for k in ("feature_mean", "feature_scale", "theta")]
        assert m.shape == s.shape == (26,) and t.shape == (26, 4)
        assert np.isfinite(m).all() and np.isfinite(s).all() and np.isfinite(t).all()
        assert np.all(s > 0) and m[0] == 0 and s[0] == 1
        mean[layer], scale[layer], theta[layer] = m, s, t
    return dict(feature_mean=mean, feature_scale=scale, theta=theta)


def candidate_source(modelsha):
    assert sha(SOURCE) == SOURCE_SHA
    source = SOURCE.read_text()
    assert source.count(OLD_MODEL_SHA) == 1
    source = source.replace(OLD_MODEL_SHA, modelsha)
    start = source.index("    bundle=fnp.load(str(path))")
    end = source.index("\ndef _corpus_ownpost_features(", start)
    source = source[:start] + '''    bundle=fnp.load(str(path))
    mean,scale,theta=[bundle[key] for key in ('feature_mean','feature_scale','theta')]
    if mean.shape!=(15,26) or scale.shape!=(15,26) or theta.shape!=(15,26,4):
        raise ValueError('Frozen normalized onpolicy head shape mismatch')
    if any(value.dtype!=fnp.float64 for value in (mean,scale,theta)):
        raise ValueError('Frozen normalized onpolicy head dtype mismatch')
    return (mean,scale,theta),digest,metadata['model_sha256']

def _corpus_normalized_delta(features,support,layer):
    mean,scale,theta=support
    return ((features-mean[layer])/scale[layer])@theta[layer]

''' + source[end:]
    assert "import numpy" not in source and "import torch" not in source
    tree = ast.parse(source)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id in {"eval", "exec", "compile"} for n in ast.walk(tree))
    assert "if riders and 0 < li < 15:" in source
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.audit_output.exists()
    assert sha(SOURCE) == SOURCE_SHA
    template = dict(version="r56-onpolicy-lawful-normalized-template", parent_body_sha256=SOURCE_SHA,
        builder_sha256=sha(Path(__file__)), feature_order=list(FEATURE_ORDER),
        private_model_required=True, required_train_indices=IDS, required_layers=list(range(1, 15)),
        frozen_provenance_required=True, component_operation_order="((f-mean)/scale)@theta",
        first0_last15_skipped=True, score_claim=False, whole_predict_parity=False,
        strict_acceptance=False, no_model_no_estimator=True)
    if args.model is None:
        assert args.expected_model_sha256 is None
        template.update(candidate_built=False, reason="no frozen full14 onpolicy model supplied")
        args.audit_output.write_text(json.dumps(template, indent=2) + "\n")
        print(json.dumps(template))
        return
    assert args.expected_model_sha256 and sha(args.model) == args.expected_model_sha256
    assert not args.output.exists()
    model = json.loads(args.model.read_text())
    support = validate_model(model)
    source = candidate_source(args.expected_model_sha256)
    args.output.mkdir()
    (args.output / "estimator.py").write_text(source)
    for name in ("LICENSE", "NOTICE.md"):
        shutil.copy2(SOURCE.parent / name, args.output / name)
    path = args.output / "coefficients.npz"
    np.savez(path, **support)
    metadata = dict(model_sha256=args.expected_model_sha256, support_sha256=sha(path),
        feature_order=list(FEATURE_ORDER), train_layers=list(range(15)), offline_all_finite=True,
        dtype="float64", normalized_arithmetic_order_preserved=True,
        private_pending_dataset_license=True, training_policy=model["training_policy"])
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    template.update(candidate_built=True, model_sha256=args.expected_model_sha256,
        support_sha256=sha(path), candidate_sha256=sha(args.output / "estimator.py"),
        scope="unaccepted experimental onpolicy normalized closure; formal marginals with unchanged mixedpairs")
    (args.output / "provenance.json").write_text(json.dumps(template, indent=2) + "\n")
    (args.output / ".gitignore").write_text("/coefficients.npz\n/coefficients.json\n")
    (args.output / "SOURCE.md").write_text(
        "# R56 experimental normalized on-policy closure\n\nFrozen model inference uses only fnp. "
        "No labels, fitting, NumPy or torch in prediction. Full14 sequential TRAIN captures and "
        "same-cycle whole prediction parity, fresh rollout and strict RPC are required before "
        "acceptance. Marginal changes leave mixed central cumulants/formal birth approximations; "
        "joint realizability is unproved. Coefficients remain private.\n")
    args.audit_output.write_text(json.dumps(template, indent=2) + "\n")
    print(json.dumps(template))


if __name__ == "__main__":
    main()
