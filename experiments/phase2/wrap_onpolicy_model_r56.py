"""Bind an existing complete R54 envelope to actual head/capture provenance.

No fitting or fabricated hashes. Reads frozen evidence, writes a new private
wrapper and small audit. Existing running experiment files are never modified.
"""

import argparse
import hashlib
import json
from pathlib import Path

from build_onpolicy_closure_template_r56 import IDS, validate_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.audit_output.exists()
    experiment = args.experiment
    source = experiment / "frozen_full24_onpolicy_local_closure_26.json"
    assert sha(source) == args.expected_model_sha256
    completion = json.loads((experiment / "completion.json").read_text())
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert completion["complete"] and completion["failures"] == 0
    model = json.loads(source.read_text())
    assert model["on_policy"] and model["train_indices"] == IDS
    assert set(model["layer_models"]) == {str(i) for i in range(1, 15)}
    chain = {r["layer"]: r for r in model["prefix_hash_chain"]}
    journal = [json.loads(line) for line in (experiment / "journal.jsonl").read_text().splitlines()]
    provenance = {}
    prior = dict(train_indices=IDS, layer_models={})
    for layer in range(1, 15):
        headpath = experiment / f"full24_head_layer{layer}.json"
        prefixpath = experiment / f"full24_prefix_through{layer}.json"
        assert sha(headpath) == chain[layer]["head_sha256"]
        assert sha(prefixpath) == chain[layer]["prefix_sha256"]
        head, prefix = json.loads(headpath.read_text()), json.loads(prefixpath.read_text())
        assert head["train_indices"] == IDS and head["layer"] == layer
        assert head["head"] == model["layer_models"][str(layer)]
        assert head["preceding_prefix_sha256"] == canonical_sha(prior)
        assert prefix["train_indices"] == IDS
        assert prefix["layer_models"] == {str(i): model["layer_models"][str(i)] for i in range(1, layer+1)}
        if layer == 1:
            # Layer1 reuses the actual earlier frozen baseline, before any head.
            assert manifest["layer1_reuses_frozen_baseline"]
            capture_records = manifest["baseline_provenance"]
        else:
            capture_records = [r for r in journal if r.get("branch") == "full24"
                               and r.get("target_layer") == layer]
            assert len(capture_records) == 24 and sorted(r["idx"] for r in capture_records) == IDS
            assert all(r["complete"] and r["prefix_sha256"] == canonical_sha(prior)
                       for r in capture_records)
        provenance[str(layer)] = dict(train_indices=IDS, target_layer=layer,
            prefix_layers=list(range(1, layer)), capture_manifest_sha256=sha(manifest_path),
            preceding_prefix_sha256=head["preceding_prefix_sha256"],
            actual_head_file_sha256=sha(headpath), actual_prefix_file_sha256=sha(prefixpath),
            capture_records_sha256=canonical_sha(capture_records),
            layer1_existing_baseline_reused=layer == 1)
        prior = prefix
    model.update(fitted_layers=list(range(1, 15)),
                 training_policy="sequential_on_policy_frozen_previous_heads",
                 head_capture_provenance=provenance,
                 original_envelope_sha256=sha(source))
    validate_model(model)
    args.output.write_text(json.dumps(model, indent=2) + "\n")
    audit = dict(version="r56-actualR54-frozenprovenance-wrapper", complete=True,
        source_sha256=sha(Path(__file__)), original_model_sha256=sha(source),
        wrapper_model_sha256=sha(args.output), manifest_sha256=sha(manifest_path),
        completion_sha256=sha(experiment / "completion.json"),
        head_capture_provenance=provenance, fitting_performed=False,
        original_evidence_modified=False, score_claim=False,
        scope="modelmetadata/provenance binding only; no candidate/wholepredict/strict acceptance")
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "head_capture_provenance"}))


if __name__ == "__main__":
    main()
