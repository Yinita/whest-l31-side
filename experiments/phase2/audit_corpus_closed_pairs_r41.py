"""Paired network uncertainty and provenance checks for completed corpus pilots."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    manifest_path, completion_path = [
        args.directory / name for name in ("manifest.json", "completion.json")
    ]
    manifest, completion = [json.loads(p.read_text()) for p in (manifest_path, completion_path)]
    assert completion["complete"]
    rows = completion["rows"]
    successes = [r for r in rows if r.get("complete")]
    failures = [r for r in rows if not r.get("complete")]
    ids = [r["idx"] for r in successes]
    assert len(ids) == len(set(ids)) and set(ids) <= set(manifest["validation_indices"])
    baseline = {}
    for line in (args.baseline_directory / "journal.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["phase"] in ("baseline", "smoke") and row.get("complete"):
            baseline[row["idx"]] = row
    for r in successes:
        b = baseline[r["idx"]]
        assert r["weights_sha256"] == b["weights_sha256"]
        assert r["h18_sha256"] == b["h18_sha256"] == manifest["h18_sha256"]
        assert r["versions"] == b["versions"]
        assert digest(Path(r["path"])) == r["payload_sha256"]
        assert digest(Path(b["path"])) == b["payload_sha256"]
    before, after = [
        np.asarray([r[name] for r in successes]) for name in ("before_raw", "after_raw")
    ]
    assert len(before) > 0
    generator = np.random.default_rng(261041)
    resamples = generator.integers(0, len(before), size=(20000, len(before)))
    ratios = after[resamples].mean(axis=1) / before[resamples].mean(axis=1)
    deltas = (after - before)[resamples].mean(axis=1)
    output = dict(
        version="corpus-closed-pair-network-audit",
        script_sha256=digest(Path(__file__)),
        manifest_sha256=digest(manifest_path),
        completion_sha256=digest(completion_path),
        model_sha256=completion["model_sha256"],
        n_success=len(successes),
        failures=failures,
        zero_failure_complete_split=set(ids) == set(manifest["validation_indices"])
        and not failures,
        mean_before_raw=float(before.mean()),
        mean_after_raw=float(after.mean()),
        aggregate_after_over_before=float(after.mean() / before.mean()),
        improved_networks=int((after < before).sum()),
        paired_network_bootstrap_ratio95=np.quantile(ratios, [0.025, 0.975]).tolist(),
        paired_network_bootstrap_mean_delta95=np.quantile(deltas, [0.025, 0.975]).tolist(),
        bootstrap_seed=261041,
        bootstrap_replicates=20000,
        per_network=[
            {k: r[k] for k in ("idx", "before_raw", "after_raw", "after_over_before")}
            for r in successes
        ],
        resource=[
            dict(
                idx=r["idx"],
                maxrss_kib=r.get("maxrss_kib"),
                elapsed_s=r.get("elapsed_s"),
                unmetered_offline_cost=r["cost"]["flops_used"],
            )
            for r in successes
        ],
        first_layer_exact_equal_checks=[
            r.get("first_layer_prediction_exact_equal") for r in successes
        ],
        limits=[
            "offline raw error; diagnostic correction unmetered",
            "bootstrap resamples networks, no neuron independence",
            "common MC reference noise retained in pair; bootstrap excludes reference sampling uncertainty",
            "no adjusted/resource/stage pass",
        ],
    )
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: output[k]
                for k in (
                    "aggregate_after_over_before",
                    "improved_networks",
                    "paired_network_bootstrap_ratio95",
                    "zero_failure_complete_split",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
