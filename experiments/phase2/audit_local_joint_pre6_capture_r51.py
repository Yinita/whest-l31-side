"""Audit compact graph capture and fixed legacy-trajectory reuse tolerances.

Only validates diagnostic payloads. No training, score or resource claim.
"""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Evidence output must be new"
    folder = args.directory
    report = json.loads((folder / "completion.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    checks = []
    assert report["complete"] and len(report["rows"]) == 24
    assert sorted(row["idx"] for row in report["rows"]) == list(range(0, 384, 16))
    for row in report["rows"]:
        assert row["complete"]
        path = Path(row["path"])
        assert digest(path) == row["payload_sha256"]
        trace = np.load(path, allow_pickle=False)
        assert trace.files == ["idx", "graph_features"]
        assert int(trace["idx"]) == row["idx"]
        graph = trace["graph_features"]
        assert graph.shape == (15, 1024, 6) and np.isfinite(graph).all()
        assert np.array_equal(graph[0], np.zeros((1024, 6)))
        assert all(a["prestate_arrays_unchanged"] for a in row["graph_capture_audit"][1:])
        passed = (
            row["prior_features26_maxabs_delta"] < 1e-3
            and row["prior_state_maxabs_delta"] < 1e-5
            and row["prior_prediction_maxabs_delta"] < 1e-5
        )
        checks.append(
            {
                k: row[k]
                for k in (
                    "idx",
                    "prior_features26_maxabs_delta",
                    "prior_state_maxabs_delta",
                    "prior_prediction_maxabs_delta",
                    "graph_max_abs_rho",
                )
            }
            | dict(
                payload_hash_pass=True,
                finite_shape_firstzero_pass=True,
                legacy_pair_absolute_gate_pass=passed,
                maxabs_per_graph_feature=np.max(abs(graph), axis=(0, 1)).tolist(),
            )
        )
    result = dict(
        version="r51-complete-compact-graph-legacy-pair-audit",
        source_sha256=digest(Path(__file__)),
        capture_source_sha256=manifest["source_sha256"],
        manifest_sha256=digest(folder / "manifest.json"),
        completion_sha256=digest(folder / "completion.json"),
        complete=report["complete"],
        n=len(checks),
        failures=report["failures"],
        thresholds=dict(
            features26_maxabs_lt=1e-3, state_maxabs_lt=1e-5, prediction_maxabs_lt=1e-5
        ),
        threshold_scope="predeclared R53 legacy reuse check; not score or strictresource",
        legacy_pair_absolute_gate_pass=all(r["legacy_pair_absolute_gate_pass"] for r in checks),
        legacy_pair_pass_count=sum(r["legacy_pair_absolute_gate_pass"] for r in checks),
        total_stored_bytes=sum(p.stat().st_size for p in folder.glob("*.npz")),
        rows=checks,
        decision="Do not silently join old26 with numerically different newgraph; no gain/stage/resource claim",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
