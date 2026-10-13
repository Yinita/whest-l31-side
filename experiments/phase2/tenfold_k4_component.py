"""Exact remaining three-index K4 orbits for the regenerated identity metric.

No target/reference data are read. Submitted helpers use flopscope operations.
The contract materializes independent dense tensors only outside charged code.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.real_k211_matrix_production_r29 import matrix_wicks


def k4_identity_power_cap(c_off, lam, metric_c, w11, w12, w21, w22):
    """sum_i distinct j,k of all three retained single-K4 pK211 terms.

    Input H4=Sym6(G tensor M), G=diag(dG)+lam*C_off, M=metric_c*I.
    On distinct i,j,k, H_iijk=metric_c*G_jk/6. Each of three
    repeated-position diagrams has coefficient 1/2. No dG survives.
    """
    # Mathematical cumulant/core inputs are symmetric; float32 transport can
    # leave small numerical asymmetry. Sym6 H4 uses symmetric G exactly.
    g = (c_off+c_off.T)*(lam/2)
    sw22 = fnp.sum(w22)-w22[:, None]-w22[None, :]
    sums = g.T@w12
    # Exclude i=j and i=k even if input G's diagonal is nonzero.
    gd = fnp.diag(g)
    sj = sums[:, None]-(w12*gd)[:, None]-g*w12[None, :]
    sk = sums[None, :]-g.T*w12[:, None]-(w12*gd)[None, :]
    value = g*w11[:, None]*w11[None, :]*sw22
    value = value+sj*w11[:, None]*w21[None, :]
    value = value+sk*w21[:, None]*w11[None, :]
    value = value*(metric_c/12)
    value = (value+value.T)/2
    fnp.fill_diagonal(value, 0)
    return value


def k4_identity_feed_center_cap(c_off, lam, metric_c, w11, w21, output_mean):
    """-2*sum_i distinct j,k output_mean_i*pK111_K4feed(i,j,k).

    Official 3 CP feed groups reduce, after removing repeated indices, to
    Sym(Q,I,y*1^T), Q=lam*diag(w11) C_off diag(w11), y=metric_c*w21/4.
    The first group and diagonal part of the third group have no distinct
    support; the second group also has no distinct support for M=metric_c*I.
    """
    q = (c_off+c_off.T)*(lam/2)*w11[:, None]*w11[None, :]
    y = w21*(metric_c/4)
    weighted_y = output_mean*y
    s = q.T@output_mean
    # Six permutations pair because Q is symmetric. Correct even for nonzero
    # diagonal Q; those entries are removed from the capped sums explicitly.
    s = s-output_mean*fnp.diag(q)
    cap = s[:, None]*y[None, :]+y[:, None]*s[None, :]
    cap = cap+q*(fnp.sum(weighted_y)-2*weighted_y[:, None]-2*weighted_y[None, :])
    cap = cap/3
    cap = (cap+cap.T)/2
    fnp.fill_diagonal(cap, 0)
    return -2*cap


def k4_identity_component(c_off, lam, metric_c, wicks, output_mean):
    """Return centered capped component before harmonic factor 6/(n+4)."""
    power = k4_identity_power_cap(c_off, lam, metric_c,
                                 wicks[(1, 1)], wicks[(1, 2)],
                                 wicks[(2, 1)], wicks[(2, 2)])
    centered_feed = k4_identity_feed_center_cap(c_off, lam, metric_c,
                                              wicks[(1, 1)], wicks[(2, 1)],
                                              output_mean)
    return power+centered_feed


def independent_tensor_contract(n, rng):
    """Explicit Sym6 H4 and all three official CP feed groups, offline only."""
    raw = rng.normal(size=(n, n))
    cov = (raw+raw.T)/2
    np.fill_diagonal(cov, 0)
    diagonal = rng.normal(size=n)
    lam, metric_c = 0.43, 2.0
    g = np.diag(diagonal)+lam*cov
    metric = np.eye(n)*metric_c
    w11, w12, w21, w22, mean = rng.normal(size=(5, n))
    wick_table = {(1, 1): w11, (1, 2): w12, (2, 1): w21, (2, 2): w22}

    # Genuine four-index tensor, independent of contracted identity formula.
    unsym = np.einsum("ij,kl->ijkl", g, metric)
    h4 = sum(unsym.transpose(perm) for perm in itertools.permutations(range(4)))/24
    one = np.ones((n, n))
    eye = np.eye(n)
    groups = [(w21[:, None]*np.diag(g)[:, None]*one,
               w11[:, None]*metric, w11[:, None]*eye/4),
              (w11[:, None]*g, w21[:, None]*eye, w11[:, None]*metric),
              (w11[:, None]*g, w11[:, None]*eye,
               w21[:, None]*np.diag(metric)[:, None]*one/4)]
    feed = np.zeros((n, n, n))
    for a, b, c in groups:
        unsym3 = np.einsum("ir,jr,kr->ijk", a, b, c)
        feed += sum(unsym3.transpose(perm) for perm in itertools.permutations(range(3)))/6
    expected_raw, expected_feed = np.zeros((n, n)), np.zeros((n, n))
    for j, k in itertools.product(range(n), repeat=2):
        if j == k:
            continue
        for i in range(n):
            if i == j or i == k:
                continue
            expected_raw[j, k] += (h4[i, i, j, k]*w22[i]*w11[j]*w11[k]
                                   +h4[i, j, j, k]*w12[i]*w21[j]*w11[k]
                                   +h4[i, j, k, k]*w12[i]*w11[j]*w21[k])/2
            expected_feed[j, k] -= 2*mean[i]*feed[i, j, k]
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
        f = {p: fnp.asarray(x) for p, x in wick_table.items()}
        c = fnp.asarray(cov)
        raw = k4_identity_power_cap(c, lam, metric_c, f[(1, 1)], f[(1, 2)],
                                   f[(2, 1)], f[(2, 2)])
        centered = k4_identity_feed_center_cap(c, lam, metric_c,
                                              f[(1, 1)], f[(2, 1)], fnp.asarray(mean))
    errors = [float(np.max(np.abs(np.asarray(got)-expected)))
              for got, expected in ((raw, expected_raw), (centered, expected_feed))]
    assert max(errors) < 5e-12, errors
    # Pure diagonal G has exactly zero all-distinct support in both orbits.
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        zero = k4_identity_component(c, 0., metric_c, f, fnp.asarray(mean))
    assert np.max(np.abs(np.asarray(zero))) == 0
    return {"width": n, "power_dense_H4_max_abs_error": errors[0],
            "feed_dense_all_three_CP_groups_max_abs_error": errors[1],
            "diagonal_G_all_distinct_component_max_abs": 0.,
            "legal_cost": context.summary_dict()}


def real_contract(meta_path):
    meta = json.loads(meta_path.read_text())
    capture = Path(meta["capture"])
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == meta["capture_sha256"]
    with np.load(capture) as src:
        state = {k: src[k].copy() for k in ("mu", "sigma", "C_off", "dG", "lam_prev", "pk1v", "METRIC_C")}
    rows = []
    for n in (8, 32, 1024):
        data = {k: x[:n, :n] if x.ndim == 2 else x[:n] if x.ndim == 1 else x
                for k, x in state.items()}
        lam, metric = float(data["lam_prev"]), float(data["METRIC_C"])
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            ws = matrix_wicks(fnp.asarray(data["mu"]).astype(fnp.float64),
                              fnp.asarray(data["sigma"]).astype(fnp.float64))
            c = fnp.asarray(data["C_off"]).astype(fnp.float64)
            mean = fnp.asarray(data["pk1v"]).astype(fnp.float64)
            raw = k4_identity_power_cap(c, lam, metric, ws[(1, 1)], ws[(1, 2)], ws[(2, 1)], ws[(2, 2)])
            centered = k4_identity_feed_center_cap(c, lam, metric, ws[(1, 1)], ws[(2, 1)], mean)
            component = (raw+centered)*(6/(n+4))
        # Scalar distinct-index oracle of H4 slice and pK111 feed; does not use
        # a benchmark reference or any candidate score.
        offline_w = {p: np.asarray(x).copy() for p, x in ws.items()}
        original = lam*data["C_off"].astype(np.float64)
        g = (original+original.T)/2
        q = g*offline_w[(1, 1)][:, None]*offline_w[(1, 1)][None, :]
        y = metric*offline_w[(2, 1)]/4
        entries = list(itertools.product(range(n), repeat=2)) if n <= 32 else [
            (i*61, (i*67+5)%n) for i in range(16)]
        expected, expected_power, expected_feed = [], [], []
        for j, k in entries:
            power, feed = 0., 0.
            if j != k:
                for i in range(n):
                    if i == j or i == k:
                        continue
                    power += metric/12*(g[j, k]*offline_w[(2, 2)][i]*offline_w[(1, 1)][j]*offline_w[(1, 1)][k]
                                        +g[i, k]*offline_w[(1, 2)][i]*offline_w[(2, 1)][j]*offline_w[(1, 1)][k]
                                        +g[i, j]*offline_w[(1, 2)][i]*offline_w[(1, 1)][j]*offline_w[(2, 1)][k])
                    feed -= 2*float(data["pk1v"][i])*(q[i, j]*y[k]+q[i, k]*y[j]+q[j, k]*y[i])/3
            expected_power.append(power)
            expected_feed.append(feed)
            expected.append((power+feed)*6/(n+4))
        error = float(np.max(np.abs(np.array([np.asarray(component)[j, k] for j, k in entries])-expected)))
        power_error = float(np.max(np.abs(np.array([np.asarray(raw)[j, k] for j, k in entries])-expected_power)))
        feed_error = float(np.max(np.abs(np.array([np.asarray(centered)[j, k] for j, k in entries])-expected_feed)))
        assert error < 1e-10*max(float(np.max(np.abs(expected))), 1e-15)+2e-12, error
        rows.append({"width": n, "power_scalar_max_abs_error": power_error,
                     "centered_feed_scalar_max_abs_error": feed_error,
                     "projected_component_scalar_max_abs_error": error,
                     "projected_component_norm": float(np.linalg.norm(np.asarray(component))),
                     "raw_power_norm": float(np.linalg.norm(np.asarray(raw))),
                     "centered_feed_norm": float(np.linalg.norm(np.asarray(centered))),
                     "relative_C_off_asymmetry_before_tensor_sym": float(np.linalg.norm(original-original.T)/max(np.linalg.norm(original), 1e-30)),
                     "legal_cost": context.summary_dict()})
    return rows, {"metadata": str(meta_path), "sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                  "capture_sha256": meta["capture_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/phase2/tenfold_k4_contract_r01.json"))
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve original evidence"
    pinned = Path("benchmark/data/official_source_93d091a4/factor_k3.py")
    assert hashlib.sha256(pinned.read_bytes()).hexdigest() == "e408aa1d8544e7d77b95dad5a548560098e7265845b7f713846ff8b165007918"
    rng = np.random.default_rng(673601)
    synthetic = [independent_tensor_contract(n, rng) for n in (4, 7, 12)]
    real_rows, capture = real_contract(args.capture) if args.capture else ([], None)
    result = {"version": "tenfold.k4.r01", "complete": True, "synthetic_contract": synthetic,
              "real_contract": real_rows, "capture": capture,
              "formula": "H4=Sym6(G tensor metric_c I), G=diag(dG)+lam C_off; cap=sum distinct i [1/2 H_iijk w22_i w11_j w11_k+1/2 H_ijjk w12_i w21_j w11_k+1/2 H_ijkk w12_i w11_j w21_k]-2 sum distinct i mean_i pK111_K4feed; project=6/(n+4)",
              "coefficients": {"each_K4_power_diagram": "1/2", "H4_identity_slice": "metric_c/6", "feed_y": "metric_c/4*w21", "mean_centering": "-2", "harmonic_projection": "6/(n+4)"},
              "official_source": "https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/93d091a4c26c042bfffa28f2e76a81bc0aba94bb/src/mlp_kprop/factor_k3.py#L419-L459",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Exact remaining single-K4 three-index orbit plus K4feed mean-centering under regenerated identity metric; O(n^2), no new cubic tensor or matrix product",
              "limitations": ["Not a complete fourth-order core: combine two-index matrix diagrams, source centering, tree centering and CovCov before projection", "Identity metric only; transported nonidentity metric requires different contractions", "Existing dG/lam regeneration remains an approximation; exact orbit contractions do not prove score benefit", "K4feed is already part of transported V29 source tensors at later layers; do not add its history twice", "No adjusted score or resource acceptance claim"],
              "integration": "At nonlinear pK stage on current transported G before births: add raw K4power and current K4feed centering once to full capped K211, with inputsource/tree/CovCov; multiply joint full cap by 6/(n+4). Do not add original proxy offdiagonal on top.",
              "no_adjusted_score": True, "not_resource_acceptance": True}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print({"output": str(args.output), "synthetic_max_error": max(max(r["power_dense_H4_max_abs_error"], r["feed_dense_all_three_CP_groups_max_abs_error"]) for r in synthetic),
           "real_rows": [{k: v for k, v in r.items() if k != "legal_cost"} for r in real_rows]}, flush=True)


if __name__ == "__main__":
    main()
