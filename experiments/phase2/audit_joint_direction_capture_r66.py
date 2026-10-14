"""Same-run R58 causal features/MC evidence audit; no join or model fitting."""

# ruff: noqa: E402
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from joint_direction_stats_r58 import joint_targets, merge


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    manifest = json.loads((args.root / "manifest.json").read_text())
    journal_text = (args.root / "journal.jsonl").read_text()
    records = [json.loads(line) for line in journal_text.splitlines()]
    baselines = [r for r in records if r.get("phase") == "baseline"]
    smoke = [r for r in records if r.get("phase") == "packaging_smoke_retained"]
    assert len(baselines) == 2 and len(smoke) == 4
    identity = {(r["network"], r["stream"], r["shard"]) for r in smoke}
    assert len(identity) == 4
    rows = []
    for row in sorted(baselines, key=lambda r: r["network"]):
        assert row["complete"] and row["reference_truth_absent"] and row["trace_removed"]
        assert sha(row["path"]) == row["payload_sha256"]
        config = next(r for r in manifest["networks"] if r["network"] == row["network"])
        assert sha(config["weights"]) == config["weight_file_sha256"]
        with np.load(config["weights"], allow_pickle=False) as bundle:
            weights = bundle["weights"]
            directions = weights[manifest["target_prelayers"]][
                :, :, manifest["canonical_output_columns"]
            ].astype(np.float64)
        with np.load(row["path"], allow_pickle=False) as bundle:
            pre = bundle["pre5"].copy()
            features = bundle["causal32"].copy()
            diagonal = bundle["diagonal4"].copy()
            layers = bundle["layers"].copy()
            available = bundle["graph_available"].copy()
            prediction = bundle["prediction"].copy()
            control = bundle["baseline_control"].copy()
        assert pre.shape == (3, 64, 5) and features.shape == (3, 64, 32)
        assert np.isfinite(features).all() and np.min(pre[..., 1]) > 0
        assert np.array_equal(features[..., 1:3], pre[..., :2])
        sigma = features[..., 3]
        assert np.max(abs(sigma**2 - pre[..., 1])) < 2e-5
        assert np.array_equal(features[..., 6], pre[..., 4])
        assert np.array_equal(features[..., 9], pre[..., 2])
        assert np.allclose(features[..., 4], pre[..., 0] / sigma, rtol=1e-6, atol=1e-7)
        assert np.allclose(features[..., 22:26], diagonal / sigma[..., None]**np.arange(1, 5),
                           rtol=1e-12, atol=1e-12)
        # Gaussian mean is causal before ownpost; fp32 source order is retained.
        gaussian = (sigma.astype(np.float32) * features[..., 6].astype(np.float32)
                    + features[..., 1].astype(np.float32) * features[..., 7].astype(np.float32))
        assert np.array_equal(features[..., 8], gaussian.astype(np.float64))
        assert np.array_equal(available, [True, True, False])
        assert np.array_equal(features[2, :, 26:], np.zeros((64, 6)))
        assert np.array_equal(prediction, control)
        assert all(r["prestate_unmodified"] and r["no_current_ownpost_feature"] for r in row["audit"])
        streams = []
        for stream in (0, 1):
            capture = next(r for r in smoke if r["network"] == row["network"] and r["stream"] == stream)
            assert sha(capture["path"]) == capture["payload_sha256"]
            assert capture["truth_reference_absent"] and capture["compute_only_no_volume_commit"]
            assert capture["samples"] == 65536 and capture["projection_f32_difference_mse"] < 1e-10
            with np.load(capture["path"], allow_pickle=False) as bundle:
                stats = {key: bundle[key].copy() for key in ("count", "mean", "m2", "m3", "m4")}
            target = joint_targets(stats, directions, 1024)
            streams.append(stats)
            # Algebraic bookkeeping on the same samples, no baseline/MC splice.
            assert np.allclose(target["projected_pre4"],
                               target["diagonal_pre4"] + target["joint_pre4"], atol=1e-12, rtol=1e-12)
        combined = merge(streams[0], streams[1])
        assert int(combined["count"]) == 131072
        rows.append(dict(network=row["network"], baseline_payload_sha256=row["payload_sha256"],
                         causal32_same_capture=True, current_ownpost_not_used=True,
                         final_graph_unavailable_explicit=True, numeric_control_maxabs=0.,
                         layers=layers.tolist(), smoke_unique_records=2,
                         same_samples_target_decomposition_passed=True))
    result = dict(
        version="r66-R58actualsamecapture-causal32-smoke-audit", complete=True,
        source_sha256=sha(Path(__file__)), manifest_sha256=sha(args.root / "manifest.json"),
        journal_snapshot_sha256=hashlib.sha256(journal_text.encode()).hexdigest(), rows=rows,
        immutable_32_feature_order="pre22 withGaussianmeanslot + previouspost4Wpower/sigmapower + currentpregraph6",
        no_cross_trace_join=True, no_fitting=True, score_claim=False,
        strict_acceptance=False, stage_accepted=False,
        scope="actualsamecapture features+uniqueMCsmoke numeric identity only; data collection may still run; no signal gate, trained algorithm or score"
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
