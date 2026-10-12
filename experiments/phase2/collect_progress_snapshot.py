"""Compact read-only experiment snapshot, no queue status promotion."""

import collections
import json
from pathlib import Path


def main():
    q = json.loads(Path("experiments/phase2/ablation_queue.json").read_text())
    print("Queue", dict(collections.Counter(t["status"] for t in q["tasks"])))
    for held in range(1, 8):
        for kind in ("paired_v29_control", "charged_candidate"):
            p = Path(f"results/phase2/l03_fold{held}_{kind}.json")
            if not p.exists() or not p.stat().st_size:
                continue
            r = json.loads(p.read_text())["results"]
            print(held, kind, r["adjusted_final_layer_score"], r["n_failed_mlps"])


if __name__ == "__main__":
    main()
