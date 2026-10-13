"""Fixed actual H18 cost4 pair/marginal consistent state-change gate, target-free."""
# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
import sys
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.tenfold_pair_grade4_contract_r26 import charged_grade4, table


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(delta, baseline):
    return {"delta_rms": float(np.sqrt(np.mean(delta**2))),
            "parent_rms": float(np.sqrt(np.mean(baseline**2))),
            "maximum_delta_abs": float(np.max(np.abs(delta))),
            "relative_delta_norm": float(np.linalg.norm(delta)/max(np.linalg.norm(baseline), 1e-30))}


def main():
    output = Path("results/phase2/tenfold_pair_grade4_actual_gate_r27.json")
    assert not output.exists()
    root = Path("results/phase2/h18_joint_li14_r25")
    meta = json.loads((root/"completion.json").read_text())
    assert meta["complete"] and meta["source_sha256"] == "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
    for item in meta["files"]:
        assert sha(Path(item["path"])) == item["sha256"]
    state = np.load(root/"pre_state.npz")
    post = np.load(root/"post_state.npz")
    inputs = {(1, 1): state["C_off"], (2, 1): state["D21"], (1, 2): state["D21"].T,
              (3,): state["D3"], (4,): state["g4row"], (3, 1): state["wk431"],
              (1, 3): state["wk431"].T, (2, 2): state["wk4m"]}
    terms, _ = table()
    mu, sigma = state["mu"].astype(np.float64), state["sigma"].astype(np.float64)
    wp = Path("benchmark/data/rpc_inputs/mlp_0.npz")
    assert sha(wp) == meta["weights_sha256"]
    perm = state["internal_to_original"]
    nextw = np.load(wp)["weights"][15][perm].T.astype(np.float64)
    started = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        delta = charged_grade4(fnp.asarray(mu), fnp.asarray(sigma),
                              {k: fnp.asarray(v.astype(np.float64)) for k, v in inputs.items()}, terms)
        dp1, dp2, dp3, dp4 = [delta[(i,)] for i in range(1, 5)]
        # cost4 adds every marginal term at that exact power-count grade. The
        # parent already includes D3^2 (cost4) for p1, so remove it once.
        parent_d3square = (fnp.asarray(state["D3"].astype(np.float64))**2/72
                          * fnp.asarray((3*mu/sigma-(mu/sigma)**3)*np.exp(-(mu/sigma)**2/2)
                                        /np.sqrt(2*np.pi)/sigma**5))
        dp1 = dp1-parent_d3square
        p1, p2, p3 = [fnp.asarray(state[f"pk{i}v"].astype(np.float64)) for i in (1, 2, 3)]
        d11 = (delta[1, 1]+delta[1, 1].T)/2
        d21power = delta[2, 1]
        d22power = (delta[2, 2]+delta[2, 2].T)/2
        # Exact recenter of already represented parent power cumulants plus
        # one grade4 increment; includes products rather than mixing means.
        p11 = fnp.asarray(state["pk11"].astype(np.float64))
        p21 = fnp.asarray(state["pk21"].astype(np.float64))
        p22 = fnp.asarray(state["pk22"].astype(np.float64))
        m1, m2 = p1+dp1, p2+dp2
        c11 = p11+d11
        c21 = p21+d21power-2*m1[:, None]*c11
        c22 = (p22+d22power+4*m1[:, None]*m1[None, :]*c11
               -2*m1[None, :]*(p21+d21power)-2*m1[:, None]*(p21+d21power).T-2*c11**2)
        for array in (c11, c21, c22):
            fnp.fill_diagonal(array, 0)
        k2 = m2-m1*m1
        k3 = p3+dp3-3*m1*m2+2*m1**3
        k4 = (fnp.asarray(state["pk4v"].astype(np.float64))+dp4-4*m1*(p3+dp3)
              -3*m2*m2+12*m1*m1*m2-6*m1**4)
        dcov = (c11+c11.T)/2-fnp.asarray(post["K11"].astype(np.float64))
        fnp.fill_diagonal(dcov, k2-fnp.asarray(post["K2v"].astype(np.float64)))
        dw = fnp.asarray(nextw)
        dvar = fnp.sum((dw@dcov)*dw, axis=1)
        dmean = dw@dp1
        newprevar = fnp.sum((dw@(dcov+fnp.asarray((post["K11"]+post["K11"].T)/2).astype(fnp.float64)))*dw, axis=1)
        del newprevar
        captured = {"delta_mean": np.asarray(dp1).copy(), "delta_p2": np.asarray(dp2).copy(),
                    "delta_K2": np.asarray(k2).copy()-post["K2v"],
                    "delta_K3": np.asarray(k3).copy()-post["K3v"],
                    "delta_K4": np.asarray(k4).copy()-post["K4v"],
                    "delta_K11": np.asarray(c11).copy()-post["K11"],
                    "delta_K21": np.asarray(c21).copy()-post["K21"],
                    "delta_K22": np.asarray(c22).copy()-post["K22"],
                    "next_delta_var": np.asarray(dvar).copy(), "next_delta_mean": np.asarray(dmean).copy()}
    rows = {k: summary(v, post[k[6:]]) for k, v in captured.items() if k.startswith("delta_K")}
    rows["delta_mean"] = summary(captured["delta_mean"], post["K1v"])
    rows["delta_p2"] = summary(captured["delta_p2"], state["pk2v"])
    # Only captured parent nextstate needed to express fixed sensitivity, not
    # official truth. Its own numerical trajectory is retained separately.
    own = np.load(root/"own_prediction.npz")["predictions"]
    result = {"complete": True, "network_id": 0, "layer": 14, "state_changes": rows,
              "next_delta_var_rms": float(np.sqrt(np.mean(captured["next_delta_var"]**2))),
              "next_delta_mean_rms": float(np.sqrt(np.mean(captured["next_delta_mean"]**2))),
              "own_prediction_final_rms": float(np.sqrt(np.mean(own[-1]**2))),
              "cost": ctx.summary_dict(), "elapsed_s": time.monotonic()-started,
              "capture_completion_sha256": sha(root/"completion.json"), "source_sha256": meta["source_sha256"],
              "script_sha256": sha(Path(__file__)), "helper_sha256": sha(Path("experiments/phase2/tenfold_pair_grade4_contract_r26.py")),
              "command": sys.argv, "scope": "Target-free model-internal cost4 pair/marginal mechanism magnitude, same H18 state, no truth/oracle/directionfit. Removes already-present p1D3square. Exact recentering may generate formally higher-grade products. No complete K3/K4 distinct-support birth, no candidate score or tenfold claim."}
    assert all(np.isfinite(v).all() for v in captured.values())
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"output": str(output), "state_changes": rows, "next_delta_var_rms": result["next_delta_var_rms"], "cost": result["cost"]["flops_used"]}, indent=2))


if __name__ == "__main__":
    main()
