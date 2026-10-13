"""Audit a nondeployable partial-diagonal MC oracle probe, not an upper bound."""

import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path("results/phase2/joint_oracle_reset_r46_independent_audit.json")
DIRECTORY = Path("results/phase2/modal_joint_oracle_reset_r46")
PARENT = Path("results/phase2/modal_joint_closure_pilot_r38_r02")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mse(left, right):
    return float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))


def main():
    assert not OUT.exists()
    completion_path = DIRECTORY / "completion.json"
    report = json.loads(completion_path.read_text())
    assert report["complete"]
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    assert manifest["indices"] == [0, 16]
    labels_path = PARENT / "research_labels.npz"
    assert digest(labels_path) == manifest["labels_sha256"]
    labels = np.load(labels_path, allow_pickle=False)
    reference = labels["final_mean"]
    repeat = labels["noise_rep_final"]
    distances = np.mean((reference - repeat) ** 2, axis=1)
    order = np.argsort(distances)
    nearest = int(labels["idx"][order[0]])
    baseline = {}
    for text in (PARENT / "journal.jsonl").read_text().splitlines():
        row = json.loads(text)
        if row["phase"] in ("baseline", "smoke") and row.get("complete"):
            baseline[row["idx"]] = row
    rows = []
    for result in report["rows"]:
        if not result.get("complete"):
            rows.append(result)
            continue
        idx = result["idx"]
        base = baseline[idx]
        assert result["weights_sha256"] == base["weights_sha256"]
        assert result["h18_sha256"] == base["h18_sha256"] == manifest["anchor_source_sha256"]
        assert result["versions"] == base["versions"]
        assert digest(Path(result["path"])) == result["payload_sha256"]
        assert digest(Path(base["path"])) == base["payload_sha256"]
        data = np.load(result["path"], allow_pickle=False)
        before = np.load(base["path"], allow_pickle=False)
        position = np.flatnonzero(labels["idx"] == idx).item()
        prediction = data["prediction"]
        raw = labels["moments"][position, :, :15].astype(np.float64)
        m1, m2, m3, m4 = raw
        variance = m2 - m1**2
        target = np.stack(
            [
                m1,
                variance,
                m3 - 3 * m1 * m2 + 2 * m1**3,
                m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4 - 3 * variance**2,
            ],
            axis=-1,
        )
        assert np.array_equal(data["after_states"][1:], target[1:])
        mean_live_rounding = float(np.max(abs(prediction[1:15] - target[1:, :, 0])))
        assert mean_live_rounding < 1e-6
        first_no_reset = float(np.max(abs(data["after_states"][0] - data["states"][0])))
        assert first_no_reset == 0
        row = dict(
            idx=idx,
            primary_before_raw=mse(before["prediction"][-1], reference[position]),
            primary_oracle_after_raw=mse(prediction[-1], reference[position]),
            raw_after_over_before=result["oracle_over_before"],
            before_layer_mean_mse=[mse(before["prediction"][layer], m1[layer]) for layer in range(15)],
            oracle_layer_mean_mse=[mse(prediction[layer], m1[layer]) for layer in range(15)],
            after_states_equal_mc_targets_layers1to14=True,
            reset_mean_live_f32_rounding_maxabs=mean_live_rounding,
            analytic_layer0_no_reset=first_no_reset == 0,
            analytic_first_output_vs_baseline_maxabs=result["first_layer_maxabs"],
            covariance_min_eigenvalue=min(
                r["min_eigenvalue"] for r in result["covariance_psd_diagnostic"]
            ),
            covariance_eigenvalues_below_minus1e_minus9=sum(
                r["negative_eigenvalues"] for r in result["covariance_psd_diagnostic"]
            ),
            elapsed_s=result["elapsed_s"],
            maxrss_kib=result["maxrss_kib"],
        )
        if idx == nearest:
            row["repeat_reference"] = dict(
                network_mapping="inferred nearest of32; README does not state network ID",
                before_raw=mse(before["prediction"][-1], repeat),
                oracle_after_raw=mse(prediction[-1], repeat),
                after_over_before=mse(prediction[-1], repeat)
                / mse(before["prediction"][-1], repeat),
            )
        rows.append(row)
    success = [r for r in rows if "primary_before_raw" in r]
    output = dict(
        version="partial-diagonal-oracle-r46-independent-audit",
        script_sha256=digest(Path(__file__)),
        source_report_sha256=digest(completion_path),
        source_manifest_sha256=digest(DIRECTORY / "manifest.json"),
        labels_sha256=digest(labels_path),
        rows=rows,
        failures=report["failures"],
        mean_before_raw=float(np.mean([r["primary_before_raw"] for r in success])),
        mean_oracle_after_raw=float(np.mean([r["primary_oracle_after_raw"] for r in success])),
        aggregate_raw_ratio=float(
            np.mean([r["primary_oracle_after_raw"] for r in success])
            / np.mean([r["primary_before_raw"] for r in success])
        ),
        repeat_vector_nearest32=[
            dict(idx=int(labels["idx"][i]), mse=float(distances[i])) for i in order[:4]
        ],
        no_noise_floor_subtracted=True,
        scientific_scope="MC oracle central marginal reset under fixed approximate mixed closure; partial-diagonal mechanism diagnostic, not universal or rigorous upper bound",
        limits=[
            "truth-guided rollout is nondeployable and not a candidate",
            "sameMC errors correlated across resetlayers and primary reference",
            "two train networks no CI or tenfold certification",
            "PSD covariance does not establish joint fourth-order moment realizability",
            "firstlayerrounding/backend differences recorded",
            "final layer not reset",
            "no legal flops/RPC/resource/stage acceptance",
        ],
    )
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: output[k]
                for k in (
                    "mean_before_raw",
                    "mean_oracle_after_raw",
                    "aggregate_raw_ratio",
                    "failures",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
