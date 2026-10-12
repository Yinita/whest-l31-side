"""L19 CPU/reference uncertainty plan, no synthetic strong reference claim."""

import json
from pathlib import Path


def main():
    width = 1024
    depth = 16
    sample_flops = 2 * width * width * depth
    target = 5.779e-10
    raw_target = target / 0.25454699648935275
    # Sensitivity grid, not asserted real output variances.
    rows = []
    for variance in (0.001, 0.01, 0.1, 1):
        n = int(variance / (0.1 * raw_target)) + 1
        rows.append(
            dict(
                assumed_neuron_variance=variance,
                samples_for_mean_variance_below_10pct_target=n,
                flops_one_reference=n * sample_flops,
                flops_two_references=2 * n * sample_flops,
            )
        )
    path = Path("results/phase2/ablation_v30_114_reference_budget_plan.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                scope="Planning only; variance grid hypothetical, actual empirical variance required",
                dense_forward_flops_per_sample=sample_flops,
                stage_raw_target_at_baseline_util=raw_target,
                rows=rows,
                protocol=[
                    "Two independent fixed seeds; batch means estimate per-neuron variance and cross-batch covariance",
                    "Record actual samples, seed, network identity, summation dtype and confidence intervals",
                    "Compare two means with variance/N1+variance/N2; strong reference must match official1e9 sample quality or demonstrate tighter uncertainty",
                    "Small CPU pilot is weak noise diagnostic only, never replaces frozen official reference or grants stage acceptance",
                ],
                conclusion="CPU budget must be measured before selecting feasible N; no reference generated, no uncertainty inferred from dtype or hypothetical variance.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
