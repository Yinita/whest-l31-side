"""Target-free audit of the r38 post-ReLU signed diagonal closure operator.

Uses the frozen H18 body, changing only the suite-shape rider flag in a
diagnostic snapshot. Tiny schedules exercise internal relabeling and pruning;
there is no corpus, reference, training, resource acceptance or accuracy claim.
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

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext

PARENT = Path("estimators/sprint_public_v37b_h18_r01/estimator.py").resolve()
SHA = "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
SNAPSHOT = Path("experiments/phase2/tenfold_joint_closure_ledger_snapshot_r38.py").resolve()
OUTPUT = Path("results/phase2/tenfold_joint_closure_ledger_audit_r38.json")


def arr(value):
    return np.asarray(value, dtype=np.float64).copy()


def main():
    assert not OUTPUT.exists()
    original = PARENT.read_text()
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == SHA
    rider = "riders = (n == 1024 and L == len(CORR_BETA))"
    assert original.count(rider) == 1
    diagnostic = original.replace(rider, "riders = True")
    assert diagnostic.replace("riders = True", rider) == original
    if SNAPSHOT.exists():
        assert SNAPSHOT.read_text() == diagnostic
    else:
        SNAPSHOT.write_text(diagnostic)
    lines = diagnostic.splitlines()

    def site(fragment):
        found = [i + 1 for i, text in enumerate(lines) if fragment in text]
        assert len(found) == 1, (fragment, found)
        return found[0]

    sites = {
        "inject": site("mu = K1v if delta is None else K1v + delta"),
        "covariance": site('fnp.copyto(legs["AP0"][k_b, 1], C)'),
        "s3": site("S21 = _zero_diag(fnp.subtract(K21, rep21"),
        "e": site("S_sep = fnp.multiply((e_b)[:, None], C_off"),
        "core": site("K4_sigma = (fnp.sum(K4v)"),
        "prune": site("if prune:"),
        "row": site("rows.append(fnp.take(mu, inv_perm)"),
        "transport_s": site("s_list.append(s_b)"),
    }
    rows = []
    for prune in (False, True):
        for operator in ("mean_only", "four_diagonal"):
            spec = importlib.util.spec_from_file_location(f"r38_{prune}_{operator}", SNAPSHOT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            n, depth = 32, 4
            # Change schedules only in memory, preserving snapshot provenance.
            module.PRUNE = prune
            module.LR0 = 2
            module.S_IN = [n, n, n - 4, n - 4, n - 8, n]
            module.S_OUT = [n, n, n, n - 4, n - 4, n - 8]
            module.S_ON = [0] * depth
            module.PRUNE_EMU = False
            weights = (
                np.random.default_rng(260138).normal(size=(depth, n, n)) * math.sqrt(0.5 / n)
            ).astype(np.float32)
            estimator = module.Estimator()
            with flops.BudgetContext(flop_budget=2**41, quiet=True):
                estimator.setup(
                    SetupContext(width=n, depth=depth, flop_budget=2**41, api_version="0.1", seed=0)
                )
            maximum, counts, captured, stored_sources = {}, {}, {}, {}
            nonidentity_permutations = 0
            aliased = []

            def check(name, got, expected, tolerance=4e-6):
                got, expected = arr(got), arr(expected)
                error = float(np.max(np.abs(got - expected)))
                assert error <= tolerance * max(float(np.max(np.abs(expected))), 1), (name, error)
                counts[name] = counts.get(name, 0) + 1
                maximum[name] = max(maximum.get(name, 0), error)

            def tracer(frame, event, argument):
                nonlocal nonidentity_permutations
                if (
                    event != "line"
                    or frame.f_code.co_filename != str(SNAPSHOT)
                    or frame.f_code.co_name != "_predict_core"
                ):
                    return tracer
                v, pos = frame.f_locals, frame.f_lineno
                li = v.get("li", -1)
                if pos == sites["transport_s"]:
                    check(
                        "newborn_diagonal_source_survives_next_transport",
                        v["s_b"],
                        stored_sources[li - 1],
                    )
                if pos == sites["inject"]:
                    assert li < depth - 1
                    pre = {
                        name: arr(v[name])
                        for name in ("K1v", "K2v", "K3v", "K4v", "pk1v", "K21", "K22")
                    }
                    aliased.append(
                        bool(np.shares_memory(np.asarray(v["K1v"]), np.asarray(v["pk1v"])))
                    )
                    assert not aliased[-1]
                    delta = np.zeros(n) if v["delta"] is None else arr(v["delta"])
                    # Arbitrary deterministic residuals exercise signs and all channels.
                    # They are not fitted and use no statistical labels.
                    pattern = np.linspace(-1, 1, n)
                    sigma = arr(v["sigma"])
                    change = np.stack(
                        [
                            0.001 * sigma * (1 + pattern),
                            -0.002 * pre["K2v"] * (1 + 0.2 * pattern),
                            0.001 * sigma**3 * pattern,
                            -0.001 * sigma**4 * pattern,
                        ],
                        axis=1,
                    )
                    if operator == "mean_only":
                        change[:, 1:] = 0
                    expected = (
                        np.stack([pre["K1v"] + delta, pre["K2v"], pre["K3v"], pre["K4v"]], axis=1)
                        + change
                    )
                    expected[:, 0] = np.maximum(expected[:, 0], 0)
                    expected[:, 1] = np.maximum(expected[:, 1], 1e-10)
                    fnp.copyto(v["K1v"], fnp.asarray((expected[:, 0] - delta).astype(np.float32)))
                    for c, name in enumerate(("K2v", "K3v", "K4v"), 1):
                        fnp.copyto(v[name], fnp.asarray(expected[:, c].astype(np.float32)))
                    check("original_pk1_unchanged_no_alias", v["pk1v"], pre["pk1v"])
                    captured[li] = dict(pre=pre, change=change, expected=expected, delta=delta)
                    inverse = np.arange(n) if v["inv_perm"] is None else np.asarray(v["inv_perm"])
                    if v["inv_perm"] is not None:
                        permutation = np.asarray(v["perm"])
                        check(
                            "inverse_permutation_exact",
                            inverse[permutation],
                            np.arange(n),
                            tolerance=0,
                        )
                        nonidentity_permutations += int(np.any(permutation != np.arange(n)))
                    # Canonical saved rows and internal live rows have inverse maps.
                    feature = np.stack(
                        [arr(v[name]) for name in ("mu", "var", "sigma", "alpha", "pk1v")], axis=1
                    )
                    scale = arr(v["sigma"])[:, None] ** np.arange(1, 5)[None, :]
                    permutation = np.argsort(inverse)
                    check(
                        "canonical_features_roundtrip",
                        feature[inverse][permutation],
                        feature,
                        tolerance=0,
                    )
                    check(
                        "canonical_states_roundtrip",
                        expected[inverse][permutation],
                        expected,
                        tolerance=0,
                    )
                    check(
                        "canonical_scales_roundtrip",
                        scale[inverse][permutation],
                        scale,
                        tolerance=0,
                    )
                    captured[li]["inverse"] = inverse
                elif li in captured:
                    current = captured[li]
                    if pos == sites["covariance"]:
                        check("output_mean_includes_delta_once", v["mu"], current["expected"][:, 0])
                        check(
                            "covariance_diagonal_rebuilt",
                            np.diag(arr(v["C"])),
                            current["expected"][:, 1],
                        )
                        check(
                            "cross_covariance_unchanged",
                            arr(v["C"]) - np.diag(np.diag(arr(v["C"]))),
                            current["pre"]["K21"] * 0 + arr(v["K11"]),
                        )
                        check("central_K21_frozen", v["K21"], current["pre"]["K21"])
                        check("central_K22_frozen", v["K22"], current["pre"]["K22"])
                    elif pos == sites["s3"]:
                        old = np.zeros(n) if v["D3_w"] is None else arr(v["D3_w"])
                        new = np.zeros(n) if v["D3_new"] is None else arr(v["D3_new"])
                        check(
                            "ordinary_M_diagonal_owns_corrected_k3_once",
                            v["S3c"],
                            current["expected"][:, 2] - old - new,
                        )
                        check(
                            "delta_k3_enters_M_once",
                            arr(v["S3c"]) - (current["pre"]["K3v"] - old - new),
                            current["change"][:, 2],
                        )
                    elif pos == sites["e"]:
                        check(
                            "original_ReLU_e_uses_original_pk1",
                            v["e_b"],
                            arr(v["W_all"])[:, estimator._i12]
                            - 2 * current["pre"]["pk1v"] * arr(v["w1"]),
                        )
                    elif pos == sites["core"]:
                        st = v["st"]
                        trace = current["pre"]["K22"].sum(axis=1)
                        expected_g = (current["expected"][:, 3] + trace) * float(st["cA"])
                        expected_g += (current["expected"][:, 3].sum() + trace.sum()) * float(
                            st["cI"]
                        )
                        check("harmonic_G_rebuilt_from_corrected_k4", v["g_prev"], expected_g)
                        check("variance_carry_corrected", v["var_prev"], current["expected"][:, 1])
                    elif pos == sites["prune"]:
                        st = v["st"]
                        expected_sigma = (
                            current["expected"][:, 3].sum() * float(st["k4_c4"])
                            + current["pre"]["K22"].sum() * float(st["k4_c22"])
                        ) * float(st["P2"])
                        check("K4_sigma_rebuilt", v["K4_sigma"], expected_sigma)
                    elif pos == sites["row"]:
                        stored_sources[li] = arr(v["newborn"][3])
                        check("newborn_s3_retains_closure_delta", stored_sources[li], v["S3c"])
                        if prune:
                            check(
                                "prune_variance_uses_corrected_diag",
                                v["cdiag_prev"],
                                current["expected"][:, 1],
                            )
                            assert not np.shares_memory(
                                np.asarray(v["S3c"]), np.asarray(v["Rr_full"])
                            )

                return tracer

            with flops.BudgetContext(flop_budget=2**41, quiet=True):
                sys.settrace(tracer)
                try:
                    prediction = arr(
                        estimator.predict(
                            SimpleNamespace(width=n, weights=[fnp.asarray(w) for w in weights]),
                            2**41,
                        )
                    )
                finally:
                    sys.settrace(None)
            assert prediction.shape == (depth, n) and np.isfinite(prediction).all()
            assert len(captured) == depth - 1
            for li, current in captured.items():
                check(
                    "returned_mean_is_canonical",
                    prediction[li],
                    current["expected"][current["inverse"], 0],
                )
            if prune:
                assert nonidentity_permutations > 0
            rows.append(
                dict(
                    prune=prune,
                    operator=operator,
                    width=n,
                    depth=depth,
                    counts=counts,
                    maximum_errors=maximum,
                    nonidentity_permutations=nonidentity_permutations,
                    K1_pk1_alias_checks=aliased,
                )
            )
    report = dict(
        version="tenfold-joint-closure-ledger-audit-r38",
        complete=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_parent_sha256=SHA,
        diagnostic_snapshot_sha256=hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        snapshot_difference="literal riders=True only; tiny pruning schedules set in memory",
        rows=rows,
        no_benchmark_or_corpus_truth=True,
        interpretation="formal signed diagonal central-cumulant closure, frozen mixed central slices; not a probability distribution",
        mean_rider="output mean is represented K1 plus inherited delta; corrected K1 stores corrected output minus inherited delta",
        limits=[
            "checks arithmetic ownership, not accuracy or full joint closure",
            "K4 remains harmonic proxy; corrected K4 diagonal is not fully retained",
            "no full1024/RPC/resource/stage claim",
            "nonlinear birth feed remains original ReLU; only ordinary-M diagonal residual rebuilt",
        ],
    )
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "maximum_error": max(max(r["maximum_errors"].values()) for r in rows),
                "rows": len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
