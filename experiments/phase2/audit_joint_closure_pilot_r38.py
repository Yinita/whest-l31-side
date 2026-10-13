"""Append-only statistical analysis for fixed corpus joint-closure experiments.

Analyses network splits and paired final errors without refitting, selecting
features, or using validation labels to change a model. First-layer noise
diagnostics use only the prespecified training networks.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

INDICES = list(range(0, 512, 16))
DATA_REV = "cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8"
PARENT_SHA = "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cumulants(raw):
    m1, m2, m3, m4 = np.moveaxis(raw, -1, 0)
    v = m2 - m1**2
    return np.stack(
        [
            m1,
            v,
            m3 - 3 * m1 * m2 + 2 * m1**3,
            m4 - 4 * m1 * m3 - 3 * m2**2 + 12 * m1**2 * m2 - 6 * m1**4,
        ],
        axis=-1,
    )


def label_audit(directory, labels):
    raw = np.moveaxis(labels["moments"][:24].astype(np.float64), 1, -1)
    target = cumulants(raw)
    m1, m2, m3 = [raw[..., i] for i in range(3)]
    derivatives = np.zeros(raw.shape[:-1] + (4, 4))
    derivatives[..., 0, 0] = 1
    derivatives[..., 1, 0], derivatives[..., 1, 1] = -2 * m1, 1
    derivatives[..., 2, 0] = -3 * m2 + 6 * m1**2
    derivatives[..., 2, 1], derivatives[..., 2, 2] = -3 * m1, 1
    derivatives[..., 3, 0] = -4 * m3 + 24 * m1 * m2 - 24 * m1**3
    derivatives[..., 3, 1] = -6 * m2 + 12 * m1**2
    derivatives[..., 3, 2], derivatives[..., 3, 3] = -4 * m1, 1
    half_ulp = np.spacing(raw.astype(np.float32)).astype(np.float64) / 2
    first_order_sensitivity = np.sum(abs(derivatives) * half_ulp[..., None, :], axis=-1)
    first_rows = []
    for position, idx in enumerate(INDICES[:24]):
        path = directory / f"baseline_idx{idx:04d}.npz"
        if not path.exists():
            continue
        trace = np.load(path, allow_pickle=False)
        sigma = trace["analytic_first_mean"] * math.sqrt(2 * math.pi)
        # Exact rectified Gaussian moments through order8 give delta-method MC
        # variance of marginal cumulants. No neuron independence is assumed.
        moments = [np.ones_like(sigma)] + [
            sigma**r * 2 ** (r / 2 - 1) * math.gamma((r + 1) / 2) / math.sqrt(math.pi)
            for r in range(1, 9)
        ]
        exact_raw = np.stack(moments[1:5], axis=-1)
        exact_target = cumulants(exact_raw)
        covariance = np.stack(
            [
                np.stack(
                    [(moments[r + s] - moments[r] * moments[s]) / 100000000 for s in range(1, 5)],
                    axis=-1,
                )
                for r in range(1, 5)
            ],
            axis=-2,
        )
        ex1, ex2, ex3 = [exact_raw[..., c] for c in range(3)]
        gradient = np.zeros((1024, 4, 4))
        gradient[:, 0, 0] = 1
        gradient[:, 1, 0], gradient[:, 1, 1] = -2 * ex1, 1
        gradient[:, 2, 0] = -3 * ex2 + 6 * ex1**2
        gradient[:, 2, 1], gradient[:, 2, 2] = -3 * ex1, 1
        gradient[:, 3, 0] = -4 * ex3 + 24 * ex1 * ex2 - 24 * ex1**3
        gradient[:, 3, 1] = -6 * ex2 + 12 * ex1**2
        gradient[:, 3, 2], gradient[:, 3, 3] = -4 * ex1, 1
        nominal = np.einsum("ncr,nrs,ncs->nc", gradient, covariance, gradient)
        mse = ((target[position, 0] - exact_target) ** 2).mean(axis=0)
        first_rows.append(
            dict(
                idx=idx,
                mse=mse.tolist(),
                nominal_mc_delta_variance=nominal.mean(axis=0).tolist(),
                ratio=(mse / nominal.mean(axis=0)).tolist(),
                baseline_trace_sha256=digest(path),
            )
        )
    return dict(
        train_indices=INDICES[:24],
        n_train_first_layer_traces=len(first_rows),
        channel_order=["mean", "variance", "kappa3", "kappa4"],
        first_layer_exact_relu_mc=first_rows,
        raw_fp32_cumulant_first_order_half_ulp_sensitivity={
            "rms_by_layer_channel": np.sqrt(
                (first_order_sensitivity**2).mean(axis=(0, 2))
            ).tolist(),
            "max_by_layer_channel": first_order_sensitivity.max(axis=(0, 2)).tolist(),
            "scope": "linearized sensitivity to storage quantization, not total MC error or certified interval",
        },
        negative_variance_count=int((target[..., 1] < 0).sum()),
        scope="train24 only; analytic noise ratios descriptive because neurons share MC draws",
        noise_rep_mapping="README gives repeated seed vector but not which network; no inferred pairing used",
    )


def paired_audit(directory):
    completion_path = directory / "completion.json"
    if not completion_path.exists():
        return dict(status="not_completed", no_validation_result_claim=True)
    completion = json.loads(completion_path.read_text())
    if not completion.get("complete"):
        return dict(status="failed", completion=completion)
    model_path = directory / "frozen_joint_ridge.json"
    assert digest(model_path) == completion["model_sha256"]
    output = {}
    for mode in ("mean_only", "formal_four_state"):
        source = directory / f"{mode}_validation.json"
        audit = json.loads(source.read_text())
        assert audit["model_sha256"] == completion["model_sha256"]
        rows = audit["rows"]
        successes = [row for row in rows if row.get("complete")]
        failures = [row for row in rows if not row.get("complete")]
        ids = [row["idx"] for row in successes]
        assert len(set(ids)) == len(ids)
        assert set(ids) <= set(INDICES[24:])
        for row in successes:
            assert row["h18_sha256"] == PARENT_SHA
            assert row["paired_weight_hash_equal"]
            assert digest(Path(row["path"])) == row["payload_sha256"]
        result = dict(
            n_success=len(successes),
            n_failure=len(failures),
            failed_rows=failures,
            validation_indices=INDICES[24:],
            successful_indices=ids,
            zero_failures_complete_split=len(successes) == 8 and not failures,
            validation_source_sha256=digest(source),
        )
        if successes:
            before = np.asarray([row["before_raw"] for row in successes])
            after = np.asarray([row["after_raw"] for row in successes])
            delta = after - before
            # Network resampling preserves each network's paired common target.
            generator = np.random.default_rng(261038)
            resamples = generator.integers(0, len(before), size=(20000, len(before)))
            ratio = after[resamples].mean(axis=1) / before[resamples].mean(axis=1)
            diff = delta[resamples].mean(axis=1)
            result.update(
                network_rows=[
                    dict(
                        idx=row["idx"],
                        before_raw=row["before_raw"],
                        after_raw=row["after_raw"],
                        ratio=row["after_over_before"],
                    )
                    for row in successes
                ],
                mean_before_raw=float(before.mean()),
                mean_after_raw=float(after.mean()),
                aggregate_after_over_before=float(after.mean() / before.mean()),
                improved_networks=int((after < before).sum()),
                ratio_paired_network_bootstrap_percentile95=np.quantile(
                    ratio, [0.025, 0.975]
                ).tolist(),
                mean_delta_paired_network_bootstrap_percentile95=np.quantile(
                    diff, [0.025, 0.975]
                ).tolist(),
                raw_floor_stage1_necessary_only=bool(after.mean() < 5.779e-9),
                bootstrap_replicates=20000,
                bootstrap_seed=261038,
            )
        output[mode] = result
    return dict(
        status="completed",
        modes=output,
        model_sha256=completion["model_sha256"],
        selection="two prespecified mechanisms, no validation tuning; shared validation implies selection bias if promoted",
        scope="offline raw MSE; NumPy correction and trace are unmetered, no adjusted/resource/stage acceptance",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    assert not arguments.output.exists()
    manifest_path = arguments.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset_revision"] == DATA_REV
    assert manifest["h18_sha256"] == PARENT_SHA
    assert manifest["train_indices"] == INDICES[:24]
    assert manifest["validation_indices"] == INDICES[24:]
    label_path = arguments.directory / "research_labels.npz"
    labels = np.load(label_path, allow_pickle=False)
    assert labels["idx"].tolist() == INDICES
    result = dict(
        version="r38-network-statistical-audit",
        script_sha256=digest(Path(__file__)),
        manifest_sha256=digest(manifest_path),
        labels_sha256=digest(label_path),
        dataset_revision=DATA_REV,
        train_label_audit=label_audit(arguments.directory, labels),
        paired_final_audit=paired_audit(arguments.directory),
    )
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "paired_status": result["paired_final_audit"]["status"],
                "train_first_layer_traces": result["train_label_audit"][
                    "n_train_first_layer_traces"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
