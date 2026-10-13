"""One frozen H18 li14 target-free state shared by pair/K4 diagnostics."""
# ruff: noqa: E402
import os
import re

removed = []
for key in tuple(os.environ):
    if re.match(r"V\d+_", key):
        removed.append(key)
        os.environ.pop(key)
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path("results/phase2/h18_joint_li14_r25")
    root.mkdir(exist_ok=False)
    source = Path("estimators/sprint_public_v37b_h18_r01/estimator.py").resolve()
    expected = "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
    assert sha(source) == expected
    wp = Path("benchmark/data/rpc_inputs/mlp_0.npz")
    assert sha(wp) == "7fb2e3f519b4a90f2c3de9e64d45ad89b7f20ffd254d887e4ba4fa5c1fdf0754"
    started = time.monotonic()
    identity = {"pid": os.getpid(), "startticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "script_sha256": sha(Path(__file__)), "source_sha256": expected, "complete": False,
                "removed_environment_keys": sorted(removed), "command": sys.argv}
    (root/"launch_manifest.json").write_text(json.dumps(identity, indent=2)+"\n")
    spec = importlib.util.spec_from_file_location("h18_joint_r25", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        estimator.setup(SetupContext(width=1024, depth=16, flop_budget=2**41,
                                     api_version="0.1", seed=3909487726))
    lines = source.read_text().splitlines()
    pretrigger = next(i for i, line in enumerate(lines, 1) if "if riders and not NO_CORR:" in line)
    posttrigger = next(i for i, line in enumerate(lines, 1) if "mu = K1v if delta is None else K1v + delta" in line)
    seen = []

    def tracer(frame, event, arg):
        if event != "line" or frame.f_code.co_filename != str(source):
            return tracer
        state = frame.f_locals
        if state.get("li") != 14:
            return tracer
        if frame.f_lineno == pretrigger:
            assert state["mode"] == 1
            names = ("mu", "var", "sigma", "alpha", "phi", "Phi", "D3", "D21", "g4row", "wk4m", "wk431",
                     "C_off", "pk1v", "pk2v", "pk3v", "pk4v", "pk11", "pk21", "pk22", "dG", "lam_prev", "METRIC_C")
            values = {k: np.asarray(state.get(k, frame.f_globals.get(k))).copy() for k in names}
            assert all(v.dtype != object for v in values.values())
            perm = np.asarray(state["perm"]) if state["inv_perm"] is not None else np.arange(1024)
            inverse = np.asarray(state["inv_perm"]) if state["inv_perm"] is not None else np.arange(1024)
            values.update(internal_to_original=perm, original_to_internal=inverse,
                          source_count=np.asarray(state["A_st"].shape[0]), source_width=np.asarray(1024),
                          selected_internal_rows=np.arange(32), selected_original_neuron_ids=perm[:32])
            np.savez(root/"pre_state.npz", **values)
            for j in range(state["A_st"].shape[0]):
                z = np.asarray(state["Z_st"][j])[:32].copy()
                feedback = np.asarray(state["Zf_st"][j])[:32].copy()
                rres, rfb = state["r"], state["rfb"]
                np.savez(root/f"source_{j:02d}.npz",
                         a=np.asarray(state["A_st"][j])[:32].copy(), p=np.asarray(state["P_st"][j])[:32].copy(),
                         z=z, ell=np.asarray(state["L_st"][j]).copy(), w2=np.asarray(state["w2b_list"][j]).copy(),
                         s=np.asarray(state["s_list"][j]).copy(), e=np.asarray(state["e_list"][j]).copy(),
                         c1=np.asarray(state["c1_list"][j]).copy(), c2=np.asarray(state["c2_list"][j]).copy(), y=z[:, rres+1],
                         f1=feedback[:, :rfb], f2=feedback[:, rfb:],
                         r1=np.asarray(state["R1T_st"][j]).copy(), r2=np.asarray(state["R2T_st"][j]).copy())
            seen.append("pre")
        elif frame.f_lineno == posttrigger:
            np.savez(root/"post_state.npz", **{k: np.asarray(state[k]).copy() for k in
                                              ("K1v", "K2v", "K3v", "K4v", "K11", "K21", "K22")})
            seen.append("post")
        return tracer

    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        mlp = SimpleNamespace(width=1024, weights=[fnp.asarray(w) for w in np.load(wp)["weights"]])
        sys.settrace(tracer)
        try:
            prediction = np.asarray(estimator.predict(mlp, 2**41)).copy()
        finally:
            sys.settrace(None)
    assert seen == ["pre", "post"] and prediction.shape == (16, 1024) and np.isfinite(prediction).all()
    assert sha(source) == expected
    np.savez(root/"own_prediction.npz", predictions=prediction)
    result = {**identity, "complete": True, "weights_sha256": sha(wp), "capture_layer": 14,
              "network_id": 0, "mlp_name": "logan-fitzgerald", "official_raw_mlp_seed": 6319981554997072999,
              "grader_setup_seed": 3909487726, "dataset_commit": "aa99830fdc09fad15407b10e8e3459d3e18bba0a",
              "elapsed_instrumented_s": time.monotonic()-started, "cost": ctx.summary_dict(),
              "versions": {k: importlib.metadata.version(k) for k in ("numpy", "flopscope", "whestbench")},
              "files": [{"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(root.glob("*.npz"))],
              "scope": "No reference/targets accessed. Same H18 call own prediction. Full pair matrices internal permutation; K3 sources only internal first32 rows/all1024birthcolumns. No truncated-network/competition resource acceptance. Raw npz kept local, not Git."}
    (root/"completion.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"complete": True, "root": str(root), "elapsed_s": result["elapsed_instrumented_s"], "flops": result["cost"]["flops_used"]}), flush=True)


if __name__ == "__main__":
    main()
