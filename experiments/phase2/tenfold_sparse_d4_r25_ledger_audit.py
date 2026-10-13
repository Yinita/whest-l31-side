"""Independent actual r25 q/X lifecycle and parent ordinary-M ledger audit.

Tiny actual predict line traces; no production changes or reference access.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext

SOURCE = Path("estimators/tenfold_sparse_d4_r25/estimator.py").resolve()
OUTPUT = Path("results/phase2/tenfold_sparse_d4_r25_ledger_audit.json")
FROZEN = "b6fe6506fe37061a9efe48326349100bae18dead466ff6d80657b8e920447438"


def densefeed(q, x):
    return (
        np.einsum("ir,jr,kr->ijk", x, q, q)
        + np.einsum("ir,jr,kr->ijk", q, x, q)
        + np.einsum("ir,jr,kr->ijk", q, q, x)
    ) / 3


def slices(t):
    d3 = np.einsum("iii->i", t)
    d21 = np.einsum("iij->ij", t).copy()
    np.fill_diagonal(d21, 0)
    return d3, d21


def main():
    assert not OUTPUT.exists()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == FROZEN
    spec = importlib.util.spec_from_file_location("sparse25ledger", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lines = SOURCE.read_text().splitlines()

    def line(fragment):
        got = [i + 1 for i, s in enumerate(lines) if fragment in s]
        assert len(got) == 1, (fragment, got)
        return got[0]

    sites = {
        "move": line("sparse_q = W@sparse_q"),
        "addprior": line("D3 = sparse_feed_d3 if D3 is None"),
        "afterprior": line("sigma = fnp.sqrt(var)"),
        "beforebirthfeed": line("if sparse_q is not None and mode == 1:"),
        "rep": line("rep21 = D21_new if D21_w is None"),
        "aftersubtract": line("e_b = W_all[:, self._i12] - 2.0 * pk1v * w1"),
        "beforeupdate": line("if sparse_q is None:"),
        "afterupdate": next(
            i + 1 for i, s in enumerate(lines) if s.strip() == "if regen:" and i > 1880
        ),
    }
    rows = []
    for n in (8, 32):
        depth = 4
        weights = (
            np.random.default_rng(2531 + n).normal(size=(depth, n, n)) * math.sqrt(2 / n)
        ).astype(np.float32)
        est = mod.Estimator()
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            est.setup(
                SetupContext(width=n, depth=depth, flop_budget=2**41, api_version="0.1", seed=0)
            )
        counts = {}
        maximum = {}
        state = {}

        def arr(v):
            return np.asarray(v).astype(np.float64).copy()

        def check(name, got, wanted, tol=2e-5):
            error = float(np.max(abs(arr(got) - wanted)))
            assert error < tol * max(float(np.max(abs(wanted))), 1), (name, error)
            counts[name] = counts.get(name, 0) + 1
            maximum[name] = max(maximum.get(name, 0), error)

        def trace(frame, event, arg):
            if (
                event != "line"
                or frame.f_code.co_filename != str(SOURCE)
                or frame.f_code.co_name != "_predict_core"
            ):
                return trace
            v = frame.f_locals
            pos = frame.f_lineno
            if pos == sites["move"]:
                post = state["post"]
                w = arr(v["W"])
                state["preq"] = w @ post["q"]
                state["prex"] = w @ post["x"]
                state["prec"] = post["c"]
                state["base3"] = arr(v["D3"])
                state["base21"] = None if v["D21"] is None else arr(v["D21"])
            elif pos == sites["addprior"]:
                q, x = arr(v["sparse_q"]), arr(v["sparse_x"])
                check("actual_q_next_rawW_once", q, state["preq"])
                check("actual_X_next_rawW_once", x, state["prex"])
                check("actual_c_static", v["sparse_c"], state["prec"])
                f = densefeed(q, x)
                s3, s21 = slices(f)
                check("actual_priorfeed_D3", v["sparse_feed_d3"], s3)
                if not v["last"]:
                    check("actual_priorfeed_D21", v["sparse_feed_d21"], s21)
                state["prior3"], state["prior21"] = s3, s21
            elif pos == sites["afterprior"] and v["sparse_q"] is not None:
                check("actual_total_D3_prior_added_once", v["D3"], state["base3"] + state["prior3"])
                if not v["last"]:
                    check(
                        "actual_total_D21_prior_added_once",
                        v["D21"],
                        state["base21"] + state["prior21"],
                    )
            elif pos == sites["beforebirthfeed"]:
                state["base_new3"] = np.zeros(n) if v["D3_new"] is None else arr(v["D3_new"])
                state["base_new21"] = arr(v["D21_new"])
            elif pos == sites["rep"] and v["sparse_q"] is not None:
                q, x, c, w1, w2 = [
                    arr(v[k]) for k in ("sparse_q", "sparse_x", "sparse_c", "w1", "w2")
                ]
                qpost = q * w1[:, None]
                xnew = 1.5 * c * w2[:, None] * q * q
                current = densefeed(qpost, xnew)
                f3, f21 = slices(current)
                check("actual_new_X_coefficient", v["sparse_new_x"], xnew)
                check("actual_Q_post_gain_once", v["sparse_post_q"], qpost)
                check("actual_newfeed_D3", v["sparse_birth_feed3"], f3)
                check("actual_newfeed_D21", v["sparse_birth_feed21"], f21)
                check("actual_birth_D3_append_once", v["D3_new"], state["base_new3"] + f3)
                check("actual_birth_D21_append_once", v["D21_new"], state["base_new21"] + f21)
                check("actual_total_D3_w_gain_once", v["D3_w"], arr(v["D3"]) * w1**3)
                check(
                    "actual_total_D21_w_gain_once",
                    v["D21_w"],
                    arr(v["D21"]) * w1[:, None] ** 2 * w1[None, :],
                )
                state["qpost"], state["xnew"], state["oldx"], state["c"], state["w1"] = (
                    qpost,
                    xnew,
                    x,
                    c,
                    w1,
                )
            elif pos == sites["aftersubtract"]:
                inherit3 = np.zeros(n) if v["D3_w"] is None else arr(v["D3_w"])
                inherit21 = np.zeros((n, n)) if v["D21_w"] is None else arr(v["D21_w"])
                new3 = np.zeros(n) if v["D3_new"] is None else arr(v["D3_new"])
                check(
                    "actual_S3c_fullfeed_repeated_dedup", v["S3c"], arr(v["K3v"]) - inherit3 - new3
                )
                expected = arr(v["K21"]) - inherit21 - arr(v["D21_new"])
                np.fill_diagonal(expected, 0)
                check("actual_S21_fullfeed_repeated_dedup", v["S21"], expected)
            elif pos == sites["beforeupdate"] and v["sparse_q"] is not None:
                state["target4"] = arr(v["K4v"])
            elif pos == sites["afterupdate"]:
                if v["li"] > 0:
                    qpost, xnew, oldx, c, w1 = [
                        state[k] for k in ("qpost", "xnew", "oldx", "c", "w1")
                    ]
                    wantq = np.concatenate([qpost, np.eye(n)], axis=1)
                    wantx = np.concatenate([oldx * w1[:, None] + xnew, np.zeros((n, n))], axis=1)
                    inherited = np.sum(qpost**4 * c, axis=1)
                    wantc = np.concatenate([c, state["target4"] - inherited])
                    check("actual_Q_gain_then_unscaled_newborn_I", v["sparse_q"], wantq)
                    check("actual_X_oldgain_newfeed_once_newborn_zero", v["sparse_x"], wantx)
                    check(
                        "actual_c_birth_residual_not_full_diagonal_doublecount",
                        v["sparse_c"],
                        wantc,
                    )
                    # Represented post-feed slices = old total feed leading gain + currentfeed.
                    oldfeed = densefeed(qpost, oldx * w1[:, None])
                    newfeed = densefeed(qpost, xnew)
                    check(
                        "actual_postfeed_full_tensor_singleinheritance",
                        densefeed(arr(v["sparse_q"]), arr(v["sparse_x"])),
                        oldfeed + newfeed,
                    )
                else:
                    check("actual_initial_Q_I", v["sparse_q"], np.eye(n))
                    check("actual_initial_X_zero", v["sparse_x"], np.zeros((n, n)))
                    check("actual_initial_c_K4v", v["sparse_c"], arr(v["K4v"]))
                state["post"] = {
                    k: arr(v[val])
                    for k, val in (("q", "sparse_q"), ("x", "sparse_x"), ("c", "sparse_c"))
                }
            return trace

        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            sys.settrace(trace)
            try:
                prediction = est.predict(
                    SimpleNamespace(
                        width=n, depth=depth, weights=[fnp.asarray(w) for w in weights]
                    ),
                    2**41,
                )
            finally:
                sys.settrace(None)
        assert np.isfinite(np.asarray(prediction)).all()
        assert counts["actual_q_next_rawW_once"] == depth - 1
        assert counts["actual_X_oldgain_newfeed_once_newborn_zero"] == depth - 2
        rows.append(
            {
                "width": n,
                "depth": depth,
                "allfinite": True,
                "actual_site_counts": counts,
                "max_abs_errors": maximum,
                "legal_predict_flops_with_validation_trace": ctx.summary_dict()["flops_used"],
            }
        )
    report = {
        "version": "tenfold.sparseD4.r25.ledgeraudit",
        "complete": True,
        "candidate_sha256": FROZEN,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
        "no_reference_or_target_access": True,
        "no_adjusted_score": True,
        "limitations": [
            "Actualgenericpredict n8/32 depth4 validateslifecycle; suitepruning/regen notactualtraced",
            "Fullfeed history represented exactly but birth retainsD4only and parent M compressed",
            "Validationtrace notresourceacceptance",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
