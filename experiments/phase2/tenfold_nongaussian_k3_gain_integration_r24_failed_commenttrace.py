"""Trace actual r24 sites at tiny widths, including confinement and newborns.

No candidate edit, target/reference read, or 1024 predict. Trace captures are
validation-only and never enter submission code or benchmark scores.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import numpy as np
from whestbench import SetupContext

from local_engine import build_mlp

SOURCE = Path("estimators/tenfold_nongaussian_k3_gain_r24/estimator.py").resolve()
OUTPUT = Path("results/phase2/tenfold_nongaussian_k3_gain_integration_r24.json")


def cp_tensor(loc):
    """Independent full represented tensor; Sym averages six permutations."""
    import itertools

    def sym(a, b, c):
        return sum(np.einsum("it,jt,kt->ijk", *x) for x in itertools.permutations((a, b, c))) / 6

    n = loc["n"]
    out = np.zeros((n, n, n), dtype=np.float64)
    for index in range(loc["A_st"].shape[0]):
        a, p, z, ell = [
            np.asarray(loc[k][index]).astype(np.float64) for k in ("A_st", "P_st", "Z_st", "L_st")
        ]
        w2, s, e, c1, c2 = [
            np.asarray(loc[k][index]).astype(np.float64)
            for k in ("w2b_list", "s_list", "e_list", "c1_list", "c2_list")
        ]
        x, y = 3 * a, a * w2
        if loc["rfb"]:
            fb = np.asarray(loc["Zf_st"][index]).astype(np.float64)
            x = x + fb[:, : loc["rfb"]] @ np.asarray(loc["R1T_st"][index]).T
            y = y + fb[:, loc["rfb"] :] @ np.asarray(loc["R2T_st"][index]).T
        m = p * s + 3 * a * e + z @ ell.T
        out += sym(x, p, y) + sym(m, p, p)
        out += sym(a * c1 + p * c2, p, z[:, loc["r"] + 1, None] * np.ones((1, n)))
    return out


def main():
    assert not OUTPUT.exists()
    spec = importlib.util.spec_from_file_location("gain24integration", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = SOURCE.read_text().splitlines()

    def line(fragment):
        found = [i + 1 for i, s in enumerate(lines) if fragment in s]
        assert len(found) == 1, (fragment, found)
        return found[0]

    sites = {
        "prewick": line("w1 = W_all[:, self._i11]"),
        "postgain": line("if mode == 1 and ka < A_st.shape[0]:"),
        "postscale": line("D3_w = D3 * gain ** 3"),
        "birth": line("# ---- new source (V1.6"),
        "pretransport": line("WD = fnp.multiply(W, (w1_prev)[None, :]"),
        "postQp": line('Yq = pool.get("yq", (nj, r_old))'),
        "postAP": line("z_side ^= 1"),
        "newborn": line("if prune and A_st is not None and not skip_src:"),
        "endbirth": line("rows.append(fnp.take(mu, inv_perm) if inv_perm is not None else mu)"),
    }
    # Duplicate swap site occurs twice; select one before newborn block.
    swaps = [
        i + 1
        for i, s in enumerate(lines)
        if 'legs["AP0"], legs["AP1"] = legs["AP1"], legs["AP0"]' in s
    ]
    sites["postZ"] = swaps[0]
    reports = []
    for confine in (False, True):
        n, depth = 32, 6
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            mlp = build_mlp(n, depth, 2442 + int(confine))
            estimator = module.Estimator()
            estimator.AGE_OLD, estimator.R_OLD = 1, (12 if confine else 384)
            estimator.AGE_OLD2, estimator.R_OLD2 = 3, 8
            estimator.setup(
                SetupContext(width=n, depth=depth, flop_budget=2**41, api_version="0.1", seed=0)
            )
        memory = {}
        checks = {}
        maxima = {}

        def check(name, actual, expected, tolerance=2e-5):
            err = float(np.max(abs(np.asarray(actual, dtype=np.float64) - expected)))
            scale = float(max(np.max(abs(expected)), 1.0))
            assert err <= tolerance * scale, (name, err, scale)
            checks[name] = checks.get(name, 0) + 1
            maxima[name] = max(maxima.get(name, 0), err)

        def trace(frame, event, arg):
            if (
                frame.f_code.co_filename != str(SOURCE)
                or frame.f_code.co_name != "_predict_core"
                or event != "line"
            ):
                return trace
            v, lineno = frame.f_locals, frame.f_lineno
            if lineno == sites["prewick"] and v["mode"] == 1:
                tensor = cp_tensor(v)
                check("actual_dslices_D3_denseCP", v["D3"], np.einsum("iii->i", tensor))
                d21 = np.array(
                    [[tensor[i, i, j] if i != j else 0 for j in range(n)] for i in range(n)]
                )
                check("actual_dslices_D21_denseCP", v["D21"], d21)
                memory["tensor"] = tensor
            elif lineno == sites["postgain"]:
                expected = np.asarray(v["w1"], dtype=np.float64)
                if v["mode"] == 1:
                    a, sigma, phi, d3, g4 = [
                        np.asarray(v[k], dtype=np.float64)
                        for k in ("alpha", "sigma", "phi", "D3", "g4row")
                    ]
                    expected = (
                        expected
                        + d3 * (a * a - 1) * phi / (6 * sigma**3)
                        + g4 * (3 * a - a**3) * phi / (24 * sigma**4)
                    )
                    memory["oldAP"] = np.asarray(v["legs4"]["AP0"]).copy()
                check("actual_gain_fixedformula", v["gain"], expected, 3e-7)
                memory["gain"] = np.asarray(v["gain"]).copy()
            elif lineno == sites["postscale"]:
                k, ka = v["A_st"].shape[0], v["ka"]
                check(
                    "actual_young_AP_gain_prescale",
                    v["legs4"]["AP0"][2 * ka : 2 * k],
                    memory["oldAP"][2 * ka : 2 * k] * memory["gain"][None, None, :, None],
                    3e-7,
                )
            elif lineno == sites["birth"] and v["mode"] == 1:
                tensor = (
                    memory["tensor"]
                    * memory["gain"][:, None, None]
                    * memory["gain"][None, :, None]
                    * memory["gain"][None, None, :]
                )
                check("actual_D3_birth_subtraction", v["D3_w"], np.einsum("iii->i", tensor))
                check(
                    "actual_D21_birth_subtraction",
                    v["D21_w"],
                    np.array(
                        [[tensor[i, i, j] if i != j else 0 for j in range(n)] for i in range(n)]
                    ),
                )
            elif lineno == sites["pretransport"]:
                memory["transport"] = {
                    k: np.asarray(v[k]).copy() for k in ("W", "w1_prev", "Z_st", "Zf_st")
                }
                memory["transport"]["AP"] = np.asarray(v["legs"]["AP0"]).copy()
                memory["transport"]["k"] = v["k"]
                memory["transport"]["Qc"] = None if v["Qc"] is None else np.asarray(v["Qc"]).copy()
            elif lineno == sites["postQp"] and v["Qp"] is not None:
                t = memory["transport"]
                check("actual_QP_join_gain", v["Qp"], t["Qc"] * t["w1_prev"][:, None])
            elif lineno == sites["postAP"]:
                t = memory["transport"]
                check("actual_WD_allgain", v["WD"], t["W"] * t["w1_prev"][None, :], 3e-7)
                check(
                    "actual_young_AP_rawW_transport",
                    v["legs"]["AP1"][v["ka"] : v["k"]],
                    t["W"][None, None] @ t["AP"][v["ka"] : v["k"]],
                )
            elif lineno == sites["postZ"]:
                t = memory["transport"]
                wd = t["W"] * t["w1_prev"][None, :]
                check("actual_Z_residual_and_feed_gain", v["Z_st"], wd[None] @ t["Z_st"])
                check("actual_Zf_feedback_gain", v["Zf_st"], wd[None] @ t["Zf_st"])
            elif lineno == sites["newborn"] and v["A_st"] is not None:
                # Newborn uses raw W, including M/feed and feedback left factors.
                t = memory.get("birthstate")
                if t is not None:
                    k = v["A_st"].shape[0] - 1
                    w = np.asarray(v["W"])
                    check("actual_newborn_A_rawW", v["A_st"][k], w @ t["a"])
                    check("actual_newborn_P_rawW", v["P_st"][k], w)
                    check("actual_newborn_Z_rawW", v["Z_st"][k], w @ t["rr"])
                    check("actual_newborn_Zf_rawW", v["Zf_st"][k], w @ t["ff"])
                    if not v["trim"]:
                        check("actual_covariance_carrier_rawW", v["C_pre"], w @ t["c"] @ w.T)
            elif lineno == sites["endbirth"]:
                memory["birthstate"] = {
                    k: np.asarray(v[x]).copy()
                    for k, x in (("a", "a_b"), ("rr", "Rr_full"), ("ff", "Ff_b"), ("c", "C"))
                }
            return trace

        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            sys.settrace(trace)
            try:
                prediction = estimator.predict(mlp, 2**41)
            finally:
                sys.settrace(None)
        assert np.isfinite(np.asarray(prediction)).all()
        assert checks["actual_D3_birth_subtraction"] == depth - 2
        if confine:
            assert checks["actual_QP_join_gain"] > 0
        reports.append(
            {
                "width": n,
                "depth": depth,
                "confinement": confine,
                "allfinite": True,
                "actual_site_counts": checks,
                "site_max_abs_error": maxima,
                "legal_predict_flops_with_trace": ctx.summary_dict()["flops_used"],
            }
        )
    report = {
        "version": "tenfold.nongaussian.K3gain.r24.integration",
        "complete": True,
        "candidate_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": reports,
        "no_reference_or_target_access": True,
        "limitations": [
            "Trace dense tensors validation-only; costs are smoke-size, not candidate1024score",
            "Suite1024pruning/regen riders not exercised",
            "No strictRPC/MSE acceptance",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
