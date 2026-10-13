"""Pair frozen r42 outputs with independently generated r41 baseline/MC labels."""

import hashlib
import json
from pathlib import Path

import numpy as np

R42 = Path("results/phase2/modal_corpus_per_layer_r42")
R41 = Path("results/phase2/modal_joint_closure_quadratic_rollout_r41")
IDS = list(range(512, 640, 16))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = R42/"paired_score.json"
    assert not output.exists()
    completed = json.loads((R42/"completion.json").read_text())
    manifest = json.loads((R42/"manifest.json").read_text())
    assert manifest["validation_indices"] == IDS
    labels = np.load(R41/"research_labels.npz", allow_pickle=False)
    assert labels["idx"].tolist() == IDS
    baseline_rows = {}
    for line in (R41/"journal.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("complete") and row.get("phase") in ["baseline", "smoke"]:
            baseline_rows[row["idx"]] = row
    rows = []
    for record in sorted(completed["rows"], key=lambda row: row.get("idx", -1)):
        if not record.get("complete"):
            rows.append(record)
            continue
        idx = record["idx"]
        baseline_path = R41/f"baseline_idx{idx:04d}.npz"
        baseline = np.load(baseline_path, allow_pickle=False)
        candidate_path = Path(record["path"])
        candidate = np.load(candidate_path, allow_pickle=False)
        assert int(baseline["idx"]) == int(candidate["idx"]) == idx
        assert digest(candidate_path) == record["payload_sha256"]
        assert digest(baseline_path) == baseline_rows[idx]["payload_sha256"]
        assert record["weights_sha256"] == baseline_rows[idx]["weights_sha256"]
        reference = labels["final_mean"][IDS.index(idx)].astype(np.float64)
        before = float(np.mean((baseline["prediction"][-1].astype(np.float64)-reference)**2))
        after = float(np.mean((candidate["prediction"][-1].astype(np.float64)-reference)**2))
        first_equal = bool(np.array_equal(candidate["prediction"][0], baseline["prediction"][0]))
        first_maxabs = float(np.max(abs(candidate["prediction"][0].astype(float)-baseline["prediction"][0].astype(float))))
        assert first_maxabs < 2e-6, first_maxabs
        rows.append(dict(idx=idx, complete=True, before_raw=before, after_raw=after,
                         ratio=after/before, first_layer_bitwise_equal=first_equal,
                         first_layer_maxabs=first_maxabs,
                         weight_sha256=record["weights_sha256"],
                         baseline_sha256=digest(baseline_path), candidate_sha256=digest(candidate_path),
                         candidate_clipped_mean=record["clipped_mean"],
                         candidate_clipped_var=record["clipped_var"],
                         elapsed_diagnostic_s=record["elapsed_s"],
                         diagnostic_flops=record["cost"]["flops_used"],
                         maxrss_kib=record["maxrss_kib"]))
    successful = [row for row in rows if row.get("complete")]
    before = float(np.mean([row["before_raw"] for row in successful]))
    after = float(np.mean([row["after_raw"] for row in successful]))
    result = dict(complete=True, rows=rows, failures=len(rows)-len(successful),
                  mean_before_raw=before, mean_after_raw=after, ratio=after/before,
                  source_sha256=digest(Path(__file__)), model_sha256=manifest["model_sha256"],
                  r42_manifest_sha256=digest(R42/"manifest.json"),
                  r41_manifest_sha256=digest(R41/"manifest.json"),
                  labels_sha256=digest(R41/"research_labels.npz"),
                  scope="Fresh8 corpusN1e8 paired diagnostic. No benchmark stage/resource acceptance; no truth in rollout.")
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ["failures", "mean_before_raw", "mean_after_raw", "ratio"]}))


if __name__ == "__main__":
    main()
