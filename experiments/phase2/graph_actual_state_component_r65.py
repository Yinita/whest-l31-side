"""Reuse frozen H18 layer14 arrays for legal graph-context parity/cost.

No reference, training, current prediction run or joint target read. Independent
NumPy diagnostics are outside candidate arithmetic and never score evidence.
"""

# ruff: noqa: E402
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from tenfold_graph_features_r50 import FEATURE_ORDER, features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent(covariance, sigma, third, fourth):
    n = len(sigma)
    s = sigma.astype(np.float64)
    rho = covariance.astype(np.float64) / np.outer(s, s)
    dleft = third.astype(np.float64) / (s[:, None]**2 * s[None, :])
    dright = third.T.astype(np.float64) / (s[:, None] * s[None, :]**2)
    k22 = fourth.astype(np.float64) / (s[:, None]**2 * s[None, :]**2)
    triangle = np.diag((rho @ rho) @ rho) / ((n - 1) * (n - 2))
    return np.column_stack([
        np.sum(rho**2, axis=1) / (n - 1),
        np.sum(rho**3, axis=1) / (n - 1), triangle,
        np.sum(rho * dleft, axis=1) / (n - 1),
        np.sum(rho * dright, axis=1) / (n - 1),
        np.sum(rho**2 * k22, axis=1) / (n - 1),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = Path("results/phase2/h18_joint_li14_r25")
    completion_path = root / "completion.json"
    completion = json.loads(completion_path.read_text())
    assert completion["complete"] and completion["capture_layer"] == 14
    source_sha = "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
    assert completion["source_sha256"] == source_sha
    assert sha("estimators/sprint_public_v37b_h18_r01/estimator.py") == source_sha
    trace_path = root / "pre_state.npz"
    file_records = completion["files"]
    expected = next(row for row in file_records if Path(row["path"]).name == trace_path.name)
    assert sha(trace_path) == expected["sha256"]
    with np.load(trace_path, allow_pickle=False) as bundle:
        arrays = [bundle[key].copy() for key in ("C_off", "sigma", "D21", "wk4m")]
        inverse = bundle["original_to_internal"].copy()
        forward = bundle["internal_to_original"].copy()
    assert np.array_equal(forward[inverse], np.arange(1024))
    hashes = [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays]
    expected_values = independent(*arrays)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
        actual = features(*[fnp.asarray(array) for array in arrays])
    values = np.asarray(actual).copy()
    assert values.shape == (1024, 6) and np.isfinite(values).all()
    errors = np.max(abs(values - expected_values), axis=0)
    assert float(errors.max()) < 1e-11
    permuted = [arrays[0][np.ix_(inverse, inverse)], arrays[1][inverse],
                arrays[2][np.ix_(inverse, inverse)], arrays[3][np.ix_(inverse, inverse)]]
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as permutation_context:
        canonical = np.asarray(features(*[fnp.asarray(array) for array in permuted])).copy()
    permutation_errors = np.max(abs(canonical - values[inverse]), axis=0)
    assert float(permutation_errors.max()) < 1e-11
    assert hashes == [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays]
    cost = context.summary_dict()
    report = dict(
        version="r65-actual1024-H18-layer14-legal-graph-component", complete=True,
        source_sha256=sha(Path(__file__)), helper_sha256=sha("experiments/phase2/tenfold_graph_features_r50.py"),
        actual_capture_completion_sha256=sha(completion_path), actual_prestate_sha256=sha(trace_path),
        h18_source_sha256=source_sha, network_id=0, width=1024, layer=14,
        data_commit=completion["dataset_commit"], weights_sha256=completion["weights_sha256"],
        feature_order=list(FEATURE_ORDER), independent_maxabs_by_feature=errors.tolist(),
        permutation_maxabs_by_feature=permutation_errors.tolist(), arrays_unmodified=True,
        maxabs_by_feature=np.max(abs(values), axis=0).tolist(), lawful_component_cost=cost,
        lawful_permutation_cost=permutation_context.summary_dict(),
        fourteen_layers_component_flops_extrapolation=14 * cost["flops_used"],
        extrapolation_is_not_measured_whole_candidate=True,
        labels_or_prediction_read=False, no_fitting=True, no_current_network_rerun=True,
        mechanism="six legal covariance/third/fourth graph-context invariants, not exact true joint targets",
        strict_acceptance=False, score_claim=False,
        scope="savedsamecall actualstate component and permutation parity/cost; independent NumPy is offline only; no whole prediction or raw/adjusted gain"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("cost")}))


if __name__ == "__main__":
    main()
