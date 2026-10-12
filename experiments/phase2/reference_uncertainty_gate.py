"""L19/L20 quantify measured pilot requirements, preserve uncertainty caveats."""

import json
from pathlib import Path

import numpy as np


def main():
    pilot = json.loads(Path("results/phase2/ablation_v30_114_reference_cpu_pilot.json").read_text())
    var = float(np.mean([r["final_neuron_variance_mean"] for r in pilot["rows"]]))
    throughput = float(np.mean([r["samples_per_s"] for r in pilot["rows"]]))
    target = 5.779e-10 / 0.25454699648935275
    rows = []
    for fraction in (0.1, 0.01):
        samples = int(np.ceil(var / (fraction * target)))
        rows.append(
            dict(
                reference_noise_fraction_of_raw_target=fraction,
                projected_samples=samples,
                projected_single_reference_days=samples / throughput / 86400,
                projected_8network_two_reference_days=16 * samples / throughput / 86400,
            )
        )
    p = Path("results/phase2/ablation_v30_114_reference_uncertainty_gate.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                pilot_variance_mean=var,
                pilot_samples_per_s=throughput,
                raw_target=target,
                rows=rows,
                observed_two_mean_difference_mse=pilot["independent_mean_difference_mse"],
                conclusion="Measured pilot yields rough planning requirements; variance/N is a model-based projection, not realized reference error or CI. Correlated neuron means and variance estimation uncertainty remain. Full1e9 quality references still absent.",
                scope="Feasibility gate only; no stage acceptance",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
