"""Score official-client RPC predictions against unchanged public targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from whestbench import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", default="benchmark/data/mini_first8")
    parser.add_argument("--local", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rpc = json.loads(Path(args.predictions).read_text())
    local = json.loads(Path(args.local).read_text())
    targets = load_dataset(args.dataset, split="mini").select_columns(
        ["mlp_name", "all_layer_means", "final_means"]
    )
    rows = []
    for item in rpc["per_mlp"]:
        index = item["mlp_index"]
        target = targets[index]
        control = local["results"]["per_mlp"][index]
        if target["mlp_name"] != item["mlp_name"] or control["mlp_name"] != item["mlp_name"]:
            raise ValueError("Network identity mismatch")
        pred = np.asarray(item["predictions"], dtype=np.float32)
        truth = np.asarray(target["all_layer_means"], dtype=np.float32)
        if pred.shape != (16, 1024) or not np.isfinite(pred).all():
            raise ValueError("Invalid prediction shape or values")
        mse = float(np.mean((pred[-1] - np.asarray(target["final_means"], dtype=np.float32)) ** 2))
        summary = item["budget_summary"]
        flops = summary["flops_used"]
        wall = summary["wall_time_s"]
        residual = summary["residual_wall_time_s"]
        valid = flops <= 2**41 and wall <= 120 and residual <= 0.4
        # Preserve an actual limit violation as a failed score, never hide it.
        if not valid:
            mse = float(np.mean(np.asarray(target["final_means"], dtype=np.float32) ** 2))
        adjusted = mse * (max(0.1, flops / (2**41)) if valid else 1.0)
        rows.append(
            {
                "mlp_index": index,
                "mlp_name": item["mlp_name"],
                "final_layer_mse": mse,
                "all_layers_mse": float(
                    np.mean(((pred if valid else np.zeros_like(pred)) - truth) ** 2)
                ),
                "adjusted_final_layer_score": adjusted,
                "flops_used": flops,
                "wall_time_s": wall,
                "residual_wall_time_s": residual,
                "client_peak_rss_kib": item["client_peak_rss_kib"],
                "limits_passed": valid,
                "local_mse_relative_delta": (mse / control["final_layer_mse"] - 1),
                "local_flops_delta": flops - control["flops_used"],
            }
        )
    result = {
        "mode": "official client/server packages, local development deployment",
        "manifest": rpc["manifest"],
        "predictions_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
        "n_mlps": len(rows),
        "adjusted_final_layer_score": float(
            np.mean([r["adjusted_final_layer_score"] for r in rows])
        ),
        "final_layer_mse": float(np.mean([r["final_layer_mse"] for r in rows])),
        "mean_compute_utilization": float(np.mean([r["flops_used"] / (2**41) for r in rows])),
        "n_failed_mlps": sum(not r["limits_passed"] for r in rows),
        "max_relative_mse_delta_vs_local": max(abs(r["local_mse_relative_delta"]) for r in rows),
        "same_flops_as_local": all(r["local_flops_delta"] == 0 for r in rows),
        "max_client_peak_rss_kib": max(r["client_peak_rss_kib"] for r in rows),
        "per_mlp": rows,
        "server_max_array_bytes": 1073741824,
        "limitation": "Tokenless development server; exact grader server cap and deployment lifecycle unknown. Shape/FLOPs/wall/residual/client memory measured; not an online grade.",
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["per_mlp", "manifest"]}, indent=2)
    )


if __name__ == "__main__":
    main()
