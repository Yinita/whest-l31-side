"""Save each completed fixed on-policy layer's concise checkpoint once.

Read-only relative to raw evidence; no fitting, model or gate changes.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    folder = args.directory
    paths = sorted(
        folder.glob("evaluations_through_layer*.json"),
        key=lambda path: int(path.stem.split("layer")[-1]),
    )
    if not paths:
        print("No completed layer checkpoint")
        return
    all_rows = json.loads(paths[-1].read_text())
    journal = [json.loads(line) for line in (folder / "journal.jsonl").read_text().splitlines()]
    for layer in sorted({row["layer"] for row in all_rows}):
        output = folder / f"layer{layer}_independent_summary_r01.json"
        if output.exists():
            continue
        held = [row for row in all_rows if row["layer"] == layer]
        assert len(held) == 24
        runtime = [
            row for row in journal if row.get("target_layer") == layer and row.get("complete")
        ]
        metric = {
            key: np.asarray([row["physical_mse"][key] for row in held]).mean(axis=0)
            for key in ("zero_current", "head_current", "h18")
        }
        result = dict(
            layer=layer,
            n_held_networks=len(held),
            physical_head_over_zero=(metric["head_current"] / metric["zero_current"]).tolist(),
            physical_head_over_h18=(metric["head_current"] / metric["h18"]).tolist(),
            physical_zero_over_h18=(metric["zero_current"] / metric["h18"]).tolist(),
            head_mean_floor_count=sum(row["clipped_mean"] for row in held),
            head_var_floor_count=sum(row["clipped_var"] for row in held),
            runtime_capture_completed=len(runtime),
            runtime_median_s=float(np.median([row["elapsed_s"] for row in runtime]))
            if runtime
            else None,
            scope="fixed method checkpoint only; no tuning or finalraw/adjusted claim",
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    print(
        json.dumps(
            dict(
                total_returned=len(journal),
                counts={
                    str(layer): sum(row.get("target_layer") == layer for row in journal)
                    for layer in range(2, 15)
                },
                failures=sum(not row["complete"] for row in journal),
            )
        )
    )


if __name__ == "__main__":
    main()
