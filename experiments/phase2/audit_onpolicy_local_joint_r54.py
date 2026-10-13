"""Audit immutable on-policy branch boundaries and saved capture/hash chain.

Can audit a running checkpoint. Never fits or changes any experiment source.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    folder = args.directory
    manifest = json.loads((folder / "manifest.json").read_text())
    ids = list(range(0, 384, 16))
    assert manifest["train_indices"] == ids
    assert manifest["ridge_lambda"] == 0.01
    assert manifest["prefix_depth_semantics"] == 16
    assert manifest["cloud_layers"] == list(range(2, 15))
    for fold in range(4):
        branch = manifest["branches"][f"fold{fold}"]
        held = ids[fold * 6 : (fold + 1) * 6]
        assert branch["held"] == held
        assert branch["train"] == [idx for idx in ids if idx not in held]
    assert manifest["branches"]["full24"] == dict(train=ids, held=[])
    journal = [json.loads(line) for line in (folder / "journal.jsonl").read_text().splitlines()]
    seen = set()
    capture_checks = []
    for row in journal:
        if not row.get("complete"):
            capture_checks.append(dict(complete=False, error=row))
            continue
        branch, layer, role, idx = [row[key] for key in ("branch", "target_layer", "role", "idx")]
        key = (branch, layer, role, idx)
        assert key not in seen
        seen.add(key)
        train = manifest["branches"][branch]["train"]
        held = manifest["branches"][branch]["held"]
        assert idx in (train if role == "train" else held)
        prefix = json.loads((folder / f"{branch}_prefix_through{layer - 1}.json").read_text())
        assert prefix["train_indices"] == train
        assert set(prefix["layer_models"]) == {str(value) for value in range(1, layer)}
        assert row["prefix_sha256"] == canonical_hash(prefix)
        assert row["prefix_head_layers"] == list(range(1, layer))
        assert row["truth_argument_received"] is False
        assert row["full_setup_depth"] == 16
        assert row["stopped_before_current_injection"] is True
        assert row["full_prediction_claimed"] is False
        assert row["raw_archive_volume"] == "whest-onpolicy-r54-raw"
        assert row["raw_archive_sha256"] == row["payload_sha256"]
        path = Path(row["path"])
        if row["local_payload_stored"]:
            assert digest(path) == row["payload_sha256"]
            trace = np.load(path, allow_pickle=False)
            assert int(trace["idx"]) == idx and int(trace["layer"]) == layer
            for name, shape in (
                ("features26", (1024, 26)),
                ("state4", (1024, 4)),
                ("scales4", (1024, 4)),
                ("inherited_delta", (1024,)),
            ):
                assert trace[name].shape == shape and np.isfinite(trace[name]).all()
            assert np.min(trace["scales4"]) > 0
            # Same captured trajectory, canonical sigma normalization of own states.
            assert (
                np.max(abs(trace["features26"][:, 22:] - trace["state4"] / trace["scales4"]))
                < 1e-12
            )
        capture_checks.append(
            dict(
                complete=True,
                branch=branch,
                layer=layer,
                role=role,
                idx=idx,
                prefix_boundary_pass=True,
                local_payload_verified=row["local_payload_stored"],
                volume_digest_declaration_matches=True,
            )
        )
    head_checks = []
    for branch, config in manifest["branches"].items():
        prior = dict(train_indices=config["train"], layer_models={})
        for layer in range(1, 15):
            head_path = folder / f"{branch}_head_layer{layer}.json"
            if not head_path.exists():
                break
            head = json.loads(head_path.read_text())
            assert head["train_indices"] == config["train"]
            assert head["layer"] == layer
            assert head["preceding_prefix_sha256"] == canonical_hash(prior)
            value = head["head"]
            assert np.asarray(value["feature_mean"]).shape == (26,)
            assert np.asarray(value["feature_scale"]).shape == (26,)
            assert np.asarray(value["theta"]).shape == (26, 4)
            assert all(np.isfinite(np.asarray(v)).all() for v in value.values())
            assert min(value["feature_scale"]) > 0
            prior["layer_models"][str(layer)] = value
            prefix_path = folder / f"{branch}_prefix_through{layer}.json"
            assert json.loads(prefix_path.read_text()) == prior
            head_checks.append(
                dict(
                    branch=branch,
                    layer=layer,
                    train_boundary_pass=True,
                    prefix_hash_chain_pass=True,
                )
            )
    completion_path = folder / "completion.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
    result = dict(
        version="r54-independent-onpolicy-boundary-audit",
        script_sha256=digest(Path(__file__)),
        manifest_sha256=digest(folder / "manifest.json"),
        immutable_worker_source_sha256=manifest["source_sha256"],
        completed=completion is not None and completion["complete"],
        returned_captures=len(journal),
        failures=[row for row in journal if not row.get("complete")],
        frozen_heads=len(head_checks),
        capture_checks=capture_checks,
        head_checks=head_checks,
        scope="boundary/hash/currentstate-integrity only; Volume bytes not re-downloaded; no accuracy/resources/stage acceptance",
    )
    if result["completed"]:
        assert len(journal) == 1560 and len(head_checks) == 70
        assert completion["failures"] == 0
        result["completion_sha256"] = digest(completion_path)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("capture_checks", "head_checks")
            }
        )
    )


if __name__ == "__main__":
    main()
