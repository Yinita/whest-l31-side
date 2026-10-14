"""Bind only a complete R61 continuation to immutable R54/R60 actual evidence.

Retains original failure. Does not fit, retry, modify sources or fabricate rows.
"""

import argparse
import hashlib
import json
from pathlib import Path

from build_onpolicy_closure_template_r56 import IDS, validate_model
from wrap_onpolicy_model_r56 import canonical_sha


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.audit_output.exists()
    experiment = args.experiment
    prior = Path("results/phase2/modal_onpolicy_local_joint_r54")
    isolated = Path("results/phase2/modal_onpolicy_isolated_r60")
    source = experiment / "frozen_full24_onpolicy_local_closure_26.json"
    assert sha(source) == args.expected_model_sha256
    completion = json.loads((experiment / "completion.json").read_text())
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original = json.loads((prior / "manifest.json").read_text())
    recovered = json.loads((isolated / "completion.json").read_text())
    assert completion["complete"] and completion["failures"] == 0
    assert completion["frozen_model_sha256"] == sha(source)
    assert manifest["continuation_from_r54"] and manifest["original_failure_retained"]
    assert manifest["continuation_parent_manifest_sha256"] == sha(prior / "manifest.json")
    assert manifest["original_failure_sha256"] == sha(prior / "failure_layer8.json")
    assert manifest["recovered_r60_completion_sha256"] == sha(isolated / "completion.json")
    assert manifest["isolated_missingrow_origin_sha256"] == canonical_sha(recovered)
    assert manifest["original_failed_source_sha256"] == original["source_sha256"]
    assert sha(Path("experiments/phase2/modal_onpolicy_local_joint_r54.py")) == original["source_sha256"]
    assert sha(Path("experiments/phase2/modal_onpolicy_continuation_r61.py")) == manifest["source_sha256"]
    assert recovered["complete"] and recovered["idx"] == 320 and recovered["target_layer"] == 8
    assert sha(Path(recovered["path"])) == recovered["payload_sha256"]
    for name, digest in manifest["through7_original_artifacts"].items():
        assert sha(prior / name) == sha(experiment / name) == digest
    original_rows = rows(prior / "journal.jsonl")
    successes = [r for r in original_rows if r.get("complete")]
    failures = [r for r in original_rows if not r.get("complete")]
    assert len(successes) == 815 and len(failures) == 1
    assert failures[0] == json.loads((prior / "failure_layer8.json").read_text())
    old_index = {canonical_sha(r): r for r in successes}
    journal = rows(experiment / "journal.jsonl")
    assert len(journal) == 1560 and all(r["complete"] for r in journal)
    identity = [(r["target_layer"], r["branch"], r["role"], r["idx"]) for r in journal]
    assert len(set(identity)) == len(identity)
    counts = {name: sum(r["origin"] == name for r in journal) for name in
              ("r54_successful_capture", "r60_isolated_recovery", "r61_compute_capture")}
    assert counts == dict(r54_successful_capture=815, r60_isolated_recovery=1,
                         r61_compute_capture=744)
    archive_records = {}
    archive_reports = {}
    for path in experiment.glob("archive_layer*.json"):
        archive = json.loads(path.read_text())
        assert archive["commit_success"]
        archive_reports[path.name] = sha(path)
        for record in archive["records"]:
            key = (record["layer"], record["branch"], record["role"], record["idx"])
            assert key not in archive_records
            archive_records[key] = record
    assert len(archive_records) == 744
    for record in journal:
        assert not record["truth_argument_received"]
        assert record["stopped_before_current_injection"] and not record["full_prediction_claimed"]
        if record["origin"] == "r54_successful_capture":
            old = old_index[record["origin_row_sha256"]]
            assert all(record[key] == value for key, value in old.items())
            assert record["origin_source_sha256"] == original["source_sha256"]
        elif record["origin"] == "r60_isolated_recovery":
            assert record["origin_row_sha256"] == canonical_sha(recovered)
            assert all(record[key] == value for key, value in recovered.items())
            assert record["origin_source_sha256"] == json.loads((isolated / "manifest.json").read_text())["script_sha256"]
        else:
            assert record["origin_source_sha256"] == manifest["source_sha256"]
            assert record["raw_archive_volume"] == manifest["raw_archive_volume"]
            assert record["raw_archive_sha256"] == record["payload_sha256"]
            archived = archive_records[(record["target_layer"], record["branch"], record["role"], record["idx"])]
            assert all(record[key] == value for key, value in archived.items())
    model = json.loads(source.read_text())
    assert model["on_policy"] and model["train_indices"] == IDS
    assert set(model["layer_models"]) == {str(i) for i in range(1, 15)}
    chain = {r["layer"]: r for r in model["prefix_hash_chain"]}
    provenance = {}
    prefix_before = dict(train_indices=IDS, layer_models={})
    for layer in range(1, 15):
        headpath = experiment / f"full24_head_layer{layer}.json"
        prefixpath = experiment / f"full24_prefix_through{layer}.json"
        assert sha(headpath) == chain[layer]["head_sha256"]
        assert sha(prefixpath) == chain[layer]["prefix_sha256"]
        head, prefix = json.loads(headpath.read_text()), json.loads(prefixpath.read_text())
        assert head["train_indices"] == IDS and head["layer"] == layer
        assert head["head"] == model["layer_models"][str(layer)]
        assert head["preceding_prefix_sha256"] == canonical_sha(prefix_before)
        assert prefix["train_indices"] == IDS
        assert prefix["layer_models"] == {str(i): model["layer_models"][str(i)] for i in range(1, layer+1)}
        if layer == 1:
            assert manifest["layer1_reuses_frozen_baseline"]
            captures = manifest["baseline_provenance"]
            assert captures == original["baseline_provenance"]
        else:
            captures = [r for r in journal if r["branch"] == "full24" and r["target_layer"] == layer]
            assert len(captures) == 24 and sorted(r["idx"] for r in captures) == IDS
            assert all(r["complete"] and r["prefix_sha256"] == canonical_sha(prefix_before)
                       for r in captures)
        provenance[str(layer)] = dict(train_indices=IDS, target_layer=layer,
            prefix_layers=list(range(1, layer)), capture_manifest_sha256=sha(manifest_path),
            preceding_prefix_sha256=head["preceding_prefix_sha256"],
            actual_head_file_sha256=sha(headpath), actual_prefix_file_sha256=sha(prefixpath),
            capture_records_sha256=canonical_sha(captures),
            layer1_existing_baseline_reused=layer == 1,
            origin_counts={origin: sum(r.get("origin") == origin for r in captures)
                           for origin in counts})
        prefix_before = prefix
    continuation = dict(parent_manifest_sha256=sha(prior / "manifest.json"),
        original_failure_sha256=sha(prior / "failure_layer8.json"),
        original_failure_preserved=True, isolated_completion_sha256=sha(isolated / "completion.json"),
        continuation_manifest_sha256=sha(manifest_path), actual_capture_origin_counts=counts,
        archive_wave_report_sha256=archive_reports,
        successful_continuation_is_not_reclassified_original_success=True)
    model.update(fitted_layers=list(range(1, 15)),
        training_policy="sequential_on_policy_frozen_previous_heads",
        head_capture_provenance=provenance, original_envelope_sha256=sha(source),
        continuation_provenance=continuation)
    validate_model(model)
    args.output.write_text(json.dumps(model, indent=2) + "\n")
    report = dict(version="r62-actualcompleteR61-provenance-wrapper", complete=True,
        source_sha256=sha(Path(__file__)), original_model_sha256=sha(source),
        wrapper_model_sha256=sha(args.output), continuation_provenance=continuation,
        head_capture_provenance=provenance, fitting_performed=False,
        original_evidence_modified=False, score_claim=False,
        scope="validated actual14heads continuationmetadata only; no wholepredict/RPC/score/stage acceptance")
    args.audit_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "head_capture_provenance"}))


if __name__ == "__main__":
    main()
