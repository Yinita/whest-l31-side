"""Read-only evidence and identifiability audit before fixed R58 acquisition.

No fitting, trace join, cloud execution or score claim. Existing array hashes
are checked, and only scalar diagnostics are written to the public report.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def population_cumulants(values):
    mean = values.mean(axis=0)
    centered = values - mean
    variance = np.mean(centered**2, axis=0)
    return np.stack(
        [mean, variance, np.mean(centered**3, axis=0),
         np.mean(centered**4, axis=0) - 3 * variance**2], axis=-1
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    prep_path = Path("results/phase2/joint_direction_preparation_r58_r04.json")
    preparation = json.loads(prep_path.read_text())
    for path, digest in preparation["source_sha256"].items():
        assert sha(path) == digest
        ast.parse(Path(path).read_text())
    for name, digest in preparation["capture_helper_hashes"].items():
        assert sha(Path("experiments/phase2") / name) == digest
    networks = []
    for record in preparation["networks"]:
        assert sha(record["weights"]) == record["weight_file_sha256"]
        assert sha(record["inputs"]) == record["inputs_sha256"]
        reference = Path(record["dataset"]) / "prepared/mini/data.arrow"
        assert sha(reference) == record["original_reference_arrow_sha256"]
        with np.load(record["weights"], allow_pickle=False) as bundle:
            weights = bundle["weights"]
            assert weights.shape == (16, 1024, 1024)
            directions = weights[preparation["target_prelayers"]][
                :, :, preparation["canonical_output_columns"]
            ]
            assert directions.shape == (3, 1024, 64)
            assert hashlib.sha256(directions.tobytes()).hexdigest() == record[
                "directions_float32_bytes_sha256"
            ]
        networks.append(dict(network=record["network"], weights_sha256=sha(record["weights"]),
                             reference_sha256=sha(reference), used_public_development=True))

    # Identical neuron marginals do not identify covariance/directional targets.
    independent = np.asarray([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    shared = np.asarray([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
    marginals = [population_cumulants(values) for values in (independent, shared)]
    assert np.array_equal(*marginals)
    directions = np.ones((2, 1))
    projected = [population_cumulants(values @ directions) for values in (independent, shared)]
    assert projected[0][0, 1] == 2 and projected[1][0, 1] == 4

    mc_path = Path("results/phase2/modal_covariance_causal_r02/completion.json")
    mc = json.loads(mc_path.read_text())
    mc_row = mc["successes"][0]
    assert sha(mc_row["path"]) == mc_row["sha256"]
    with np.load(mc_row["path"], allow_pickle=False) as bundle:
        assert set(bundle.files) == {"mean", "m2", "m3", "m4", "count"}
        assert bundle["mean"].shape == (2, 16, 1024)
    # This particular sufficient-statistics archive cannot reconstruct joint
    # moments, even though the raw samples originally came from one full net.

    graph_root = Path("results/phase2/modal_local_joint_pre6_capture_r51")
    graph = json.loads((graph_root / "completion.json").read_text())
    graph_manifest = json.loads((graph_root / "manifest.json").read_text())
    assert graph["complete"] and graph["failures"] == 0 and len(graph["rows"]) == 24
    maxima = np.zeros(6)
    passed = 0
    for row in graph["rows"]:
        assert sha(row["path"]) == row["payload_sha256"]
        with np.load(row["path"], allow_pickle=False) as bundle:
            assert set(bundle.files) == {"idx", "graph_features"}
            assert bundle["graph_features"].shape == (15, 1024, 6)
            assert np.isfinite(bundle["graph_features"]).all()
            maxima = np.maximum(maxima, np.max(abs(bundle["graph_features"]), axis=(0, 1)))
        assert all(a["prestate_arrays_unchanged"] for a in row["graph_capture_audit"][1:])
        passed += (row["prior_features26_maxabs_delta"] < 1e-3 and
                   row["prior_state_maxabs_delta"] < 1e-5 and
                   row["prior_prediction_maxabs_delta"] < 1e-5)
    assert passed == 14
    source = Path("experiments/phase2/modal_local_joint_pre6_capture_r51.py").read_text()
    assert 'np.savez_compressed(compact, idx=np.asarray(row["idx"]), graph_features=graph)' in source
    # Same-run local32 existed in transport, but its bytes were not retained.
    # A transport digest alone cannot recover discarded feature/state arrays.
    worker = Path("experiments/phase2/modal_joint_direction_mc_r58_r02.py")
    tree = ast.parse(worker.read_text())
    imports = [ast.unparse(node) for node in tree.body
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("joint_direction_config" in value for value in imports)
    old_failure = Path("results/phase2/joint_direction_mc_r58/failure_import_r01.json")
    failure = json.loads(old_failure.read_text())
    assert not failure["complete"] and failure["MC_records"] == 0
    contract_path = Path("results/phase2/joint_direction_contract_reaudit_r64_r01.json")
    contract = json.loads(contract_path.read_text())
    assert contract["complete"] and contract["no_cloud"]

    report = dict(
        version="r64-joint-direction-graph-reuse-identifiability-audit", complete=True,
        source_sha256=sha(Path(__file__)), preparation_sha256=sha(prep_path),
        immutable_r58_source_hashes=preparation["source_sha256"], networks=networks,
        mathematical_contract_sha256=sha(contract_path),
        identifiability=dict(equal_marginal_cumulants=True,
                             directional_variance_independent=2., directional_variance_shared=4.,
                             conclusion="marginal-only archives cannot identify joint contractions"),
        existing_MC=dict(completion_sha256=sha(mc_path), original_successes=len(mc["successes"]),
                         original_failures=len(mc["failures"]), inspected_payload_sha256=mc_row["sha256"],
                         stored_shape=[2, 16, 1024], raw_samples_saved=False,
                         joint_labels_recoverable=False, no_MC_repeated=True),
        existing_graph=dict(manifest_sha256=sha(graph_root / "manifest.json"),
                            completion_sha256=sha(graph_root / "completion.json"),
                            capture_source_sha256=graph_manifest["source_sha256"], networks=24,
                            failures=0, preserved_arrays=["idx", "graph_features"],
                            original_gate_pass_count=14, gate_not_relaxed=True,
                            maxabs_per_feature=maxima.tolist(), transport_local32_discarded=True,
                            no_cross_trace_join=True, fitting_performed=False),
        deployment=dict(original_import_failure_sha256=sha(old_failure), original_failure_retained=True,
                        immutable_r02_inlines_configuration=True, new_MC_results_available=False),
        mechanism="measure true K2/K3/K4 next-weight contractions before choosing joint-context algorithm",
        minimum_next_experiment="fixed four unique MCsmoke shards plus two sameworker H18 capture/control; reuse smoke on integrity pass; fixed remaining252 only after numeric/data gates",
        gain_claim=False, stage_accepted=False, scope="read-only existing-evidence reuse audit; no fitted model, candidate, score or generalization"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
