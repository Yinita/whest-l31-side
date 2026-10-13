"""Shape-only analytic main-GEMM cost model; no array evaluation or score."""

import hashlib
import json
from pathlib import Path


def main():
    output = Path("results/phase2/tenfold_alllayers_analytic_cost_r01.json")
    assert not output.exists()
    width, r1, r2, age1, age2 = 1024, 384, 224, 4, 7
    budget = 2**41
    rows = []
    # flopscope conventional2*m*k*n times float64 precision factor2.
    unit = 4*width**3
    for layer in range(1, 15):
        sources = layer
        old = max(0, layer-age1)
        tier2 = max(0, layer-age2)
        tier1 = old-tier2
        young = sources-old
        # 3 fused lower-support+1 CovCov+5 hypertree mean-centering GEMMs.
        fixed = 9*unit
        dense_ha_hp = 2*young*unit
        shared_ha_hp = 2*4*width**2*(tier1*r1+tier2*r2)
        shared_hq = 4*width**2*((r1 if tier1 else 0)+(r2 if tier2 else 0))
        # Source_projection: x1/y1 and Hx1/Hy1 require six n²*rfb GEMMs;
        # M/HM requires three n²*(rres+2) GEMMs. Thin ranks8 and18.
        thin = sources*4*width**2*(6*8+3*18)
        total = fixed+dense_ha_hp+shared_ha_hp+shared_hq+thin
        rows.append({"layer": layer, "source_slots": sources, "young": young,
                     "tier1": tier1, "tier2": tier2,
                     "fixed_gemm_flops": fixed, "dense_source_gemm_flops": dense_ha_hp,
                     "shared_source_gemm_flops": shared_ha_hp, "shared_hq_flops": shared_hq,
                     "thin_source_gemm_flops": thin, "gemm_subtotal": total})
    extra = sum(row["gemm_subtotal"] for row in rows)
    report = {"version": "tenfold.alllayers.analytic.r01", "rows": rows,
              "main_gemm_extra_flops": extra, "extra_budget_fraction": extra/budget,
              "approx_parent_flops": 450_000_000_000,
              "approx_parent_plus_extra_gemm": 450_000_000_000+extra,
              "budget": budget,
              "hypothesis": "All li1..14 complete production to scalar regeneration; no layer selection from labels",
              "not_actual_resource_acceptance": True, "no_adjusted_score": True,
              "exclusions": ["Elementwise/Wick/astype/source-column reductions and K31 costs",
                             "Updated trajectory changing parent rangefinder/prune op stream",
                             "Wall/residual and memory; do not infer120s or0.4s pass"],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, indent=2)+"\n")
    print({k: v for k, v in report.items() if k not in ("rows", "exclusions")})


if __name__ == "__main__":
    main()
