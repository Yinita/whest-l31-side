"""Target-free allsource mapping and full-rank contraction cost gate at layer14."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from full_source_cp_mapping_contract import full_source_factors
from real_source_full_k211 import KEYS, independent_v29_slices
from real_source_k211_contract import direct_entry
from source_cap_three_products import capped_three_products


def main():
    output = Path("results/phase2/ablation_v30_067_r19_all_source_k211.json")
    assert not output.exists()
    meta_path = Path("results/phase2/ablation_v30_067_r18_all_source_capture.json")
    meta = json.loads(meta_path.read_text())
    state_path = Path(meta["capture"])
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == meta["capture_sha256"]
    state = np.load(state_path)
    source_rows, oracle = [], np.zeros(16)
    entries = [(i*61, (i*67+5)%1024) for i in range(16)]
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        sigma, alpha, mean = map(fnp.asarray, (state["sigma"], state["alpha"], state["pk1v"]))
        gate = flops.stats.norm.cdf(alpha).astype(fnp.float32)
        density = flops.stats.norm.pdf(alpha).astype(fnp.float32)
        u = 2*sigma*(density+alpha*gate)-2*mean*gate
        total = fnp.zeros((1024, 1024), dtype=fnp.float32)
        d3sum, d21sum = fnp.zeros(1024, dtype=fnp.float32), fnp.zeros((1024, 1024), dtype=fnp.float32)
        for path, sha in meta["source_files_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
            data = np.load(path)
            values = [data[k].copy() for k in KEYS]
            before = ctx.summary_dict()["flops_used"]
            cp = full_source_factors(*map(fnp.asarray, values))
            a, b, c = cp
            capped = capped_three_products(a, b, c, u, gate)
            total = total+capped
            component_flops = ctx.summary_dict()["flops_used"]-before
            d3 = fnp.sum(a*b*c, axis=1)
            d21 = ((a*b)@c.T+(a*c)@b.T+(b*c)@a.T)/3
            fnp.fill_diagonal(d21, 0)
            d3sum, d21sum = d3sum+d3, d21sum+d21
            source_rows.append({"source_file": path, "cp_rank": a.shape[1],
                                "component_flops": component_flops})
            # Offline independent V29 and scalar checks are outside submit logic.
            # Their CPUtime is diagnostic and no resource pass is inferred.
            exact3, exact21 = independent_v29_slices([x.astype(np.float64) for x in values])
            source_rows[-1].update(D3_relative_error=float(np.linalg.norm(np.asarray(d3)-exact3)/max(np.linalg.norm(exact3), 1e-30)),
                                   D21_relative_error=float(np.linalg.norm(np.asarray(d21)-exact21)/max(np.linalg.norm(exact21), 1e-30)))
            af, bf, cf = [np.asarray(x).astype(np.float64) for x in cp]
            uf, vf = np.asarray(u).astype(np.float64), np.asarray(gate).astype(np.float64)
            oracle += [direct_entry(af, bf, cf, uf, vf, j, k) for j, k in entries]
    got = np.asarray(total).astype(np.float64)
    actual = np.array([got[j, k] for j, k in entries])
    oracle_error = float(np.max(np.abs(actual-oracle)))
    assert oracle_error < 5e-5*max(float(np.max(np.abs(oracle))), 1e-12)+1e-8
    d3rel = float(np.linalg.norm(np.asarray(d3sum)-state["D3"])/np.linalg.norm(state["D3"]))
    d21rel = float(np.linalg.norm(np.asarray(d21sum)-state["D21"])/np.linalg.norm(state["D21"]))
    assert max(d3rel, d21rel) < 5e-5, (d3rel, d21rel)
    report = {"version": "v30.067.r19", "complete": True, "source_rows": source_rows,
              "source_slots": len(source_rows), "source_columns_each": 1024,
              "allsource_D3_relative_error_vs_actual_state": d3rel,
              "allsource_D21_relative_error_vs_actual_state": d21rel,
              "fixed16_capped_oracle_max_error": oracle_error,
              "component_flops_sum": sum(r["component_flops"] for r in source_rows),
              "legal_diagnostic_cost": ctx.summary_dict(),
              "projected_single_orbit_core_norm": float(np.linalg.norm(got)*6/1028),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "capture_metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
              "before": "One full real source verified; other slots unknown",
              "after": "All14 actual represented sources jointly match captured D3/D21; one complete K111-to-K211 orbit costed",
              "scope": "Target-free layer14 fullrank diagnostic; other K31/K211 diagrams and proxy removal pending",
              "no_adjusted_score": True, "not_resource_acceptance": True}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({k: v for k, v in report.items() if k not in ("source_rows", "legal_diagnostic_cost")})


if __name__ == "__main__":
    main()
