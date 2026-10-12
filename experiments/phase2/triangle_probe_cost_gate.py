"""H31 bounded budget vs all-output Hutchinson probe count."""

import json
from pathlib import Path


def main():
    n = 1024
    budget = 2**41
    baseline = 559754764911
    per_probe = 6 * n**3
    rows = []
    for probes in (1, 4, 16, 64, 128, 256):
        cost = probes * per_probe
        rows.append(
            dict(
                probes=probes,
                extra_flops=cost,
                total_projected=baseline + cost,
                fits_full_budget=baseline + cost <= budget,
            )
        )
    p = Path("results/phase2/ablation_v30_061_probe_cost_gate.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                rows=rows,
                max_probe_count_from_remaining_budget=(budget - baseline) // per_probe,
                conclusion="Three dense matvec per output/probe gives6sn³ FLOPs before elementwise/probe generation; maximum here is optimistic. Variance formula determines whether this count can resolve actual signal; no actual deep-state signal measured.",
                scope="Analytic remaining-budget gate, not measured runtime",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
