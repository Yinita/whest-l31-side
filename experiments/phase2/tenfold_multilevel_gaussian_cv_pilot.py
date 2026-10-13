"""Target-free multilevel Gaussian-linear innovation control variate pilot.

Research-only NumPy; source traces supply features only, never reference.
For frozen mu, gain and Gaussian-linear A, every local innovation is centered
by its exact Gaussian ReLU expectation. Recursion preserves E[control]=0.
Train-only beta then gives unbiased E[f-beta*control] in exact arithmetic.
"""

# ruff: noqa: E402
import os

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def gaussian_relu_mean(mu, sigma):
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    safe = np.maximum(sigma, 1e-30)
    alpha = mu / safe
    cdf = np.array([0.5 * (1.0 + math.erf(float(a) / math.sqrt(2))) for a in alpha])
    pdf = np.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi)
    value = mu * cdf + safe * pdf
    return np.where(sigma > 0, value, np.maximum(mu, 0))


def build_model(weights, mu, gain):
    """A rows are input coordinates; exported API weight has row-forward h@w."""
    depth, n, _ = weights.shape
    a = np.empty_like(weights, dtype=np.float64)
    a[0] = weights[0]
    for layer in range(1, depth):
        a[layer] = (a[layer - 1] * gain[layer - 1][None, :]) @ weights[layer]
    sigma = np.sqrt(np.sum(a * a, axis=1))
    means = np.stack([gaussian_relu_mean(mu[layer], sigma[layer]) for layer in range(depth)])
    assert a.shape == (depth, n, n) and means.shape == (depth, n)
    return a, sigma, means


def sample_outputs(weights, mu, gain, a, means, count, seed):
    rng = np.random.default_rng(seed)
    depth, n, _ = weights.shape
    forward_rows, control_rows = [], []
    for offset in range(0, count, 128):
        x = rng.standard_normal((min(128, count - offset), n))
        h, control = x.copy(), x.copy()
        for layer in range(depth):
            h = np.maximum(h @ weights[layer], 0)
            fluct = x @ a[layer]
            innovation = np.maximum(mu[layer] + fluct, 0) - means[layer] - gain[layer] * fluct
            control = (control @ weights[layer]) * gain[layer] + innovation
        forward_rows.append(h)
        control_rows.append(control)
    return np.concatenate(forward_rows), np.concatenate(control_rows)


def antithetic_outputs(weights, count, seed):
    assert count % 2 == 0
    rng = np.random.default_rng(seed)
    paired = []
    for offset in range(0, count // 2, 128):
        x = rng.standard_normal((min(128, count // 2 - offset), weights.shape[1]))
        h = np.concatenate([x, -x])
        for w in weights:
            h = np.maximum(h @ w, 0)
        half = len(h) // 2
        paired.append((h[:half] + h[half:]) / 2)
    return np.concatenate(paired)


def scientific_cost(n, depth, train_count, *, dtype_rate):
    # This transparent multiply+add estimate is not flopscope measured billing.
    dense = n * n * (2 * n - 1)
    build = dtype_rate * ((depth - 1) * dense + (depth - 1) * n**2 + 2 * depth * n**2)
    raw_per = dtype_rate * (depth * n * (2 * n - 1) + depth * n)
    # Per layer: actual hW, fixed Gaussian xA, recursive controlW = three matmuls.
    cv_per = dtype_rate * (3 * depth * n * (2 * n - 1) + 9 * depth * n)
    budget = 2**41
    raw_count = budget // raw_per
    cv_held = max(0, (budget - build) // cv_per - train_count)
    return {"dtype_billing_rate": dtype_rate, "budget": budget, "coefficient_build_flops": int(build),
            "raw_sample_flops": int(raw_per), "cv_sample_flops": int(cv_per),
            "optimistic_full_budget_iid_samples": int(raw_count),
            "optimistic_cv_held_samples_after_pilot": int(cv_held),
            "cv_to_raw_sample_cost_ratio": cv_per / raw_per,
            "omitted": ["all data movement/RNG and sample centered variance/beta calculation",
                        "PDF/CDF expectation costs and Python/RPC overhead",
                        "actual implementation may fuse or reuse exact intermediates"]}


def self_check():
    rng = np.random.default_rng(261013701)
    weights = rng.standard_normal((3, 5, 5)) / math.sqrt(5)
    mu = rng.standard_normal((3, 5))
    mu[0] = 0
    gain = rng.uniform(0.1, 0.9, (3, 5))
    a, sigma, means = build_model(weights, mu, gain)
    np.testing.assert_array_equal(a[0], weights[0])
    np.testing.assert_allclose(means[0], sigma[0] / math.sqrt(2 * math.pi), atol=1e-14)
    # One layer has exact centered control, no approximate internal distribution.
    y, g = sample_outputs(weights[:1], mu[:1], gain[:1], a[:1], means[:1], 16, 261013702)
    np.testing.assert_allclose(g, y - means[0], atol=1e-14)
    # Zero matrices reduce to constant ReLU; centered innovations vanish.
    z = np.zeros_like(weights)
    az, _, ez = build_model(z, mu, gain)
    _, gz = sample_outputs(z, mu, gain, az, ez, 8, 261013703)
    np.testing.assert_array_equal(gz, np.zeros_like(gz))
    asymmetric = np.array([[[1., 2.], [3., 4.]]])
    np.testing.assert_array_equal(np.array([[1., 0.]]) @ asymmetric[0], [[1., 2.]])
    return {"single_relu_control_equals_actual_minus_exact_mean": True,
            "zero_gaussian_variance_centering": True, "api_row_forward_h_at_w": True,
            "all_controls_known_zero_expectation_by_induction": "analytic, no numerical reference used"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights")
    parser.add_argument("--features")
    parser.add_argument("--output")
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--held", type=int, default=512)
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    checks = self_check()
    if args.self_check_only:
        print(json.dumps(checks))
        return
    assert args.weights and args.features and args.output and args.train > 1 and args.held > 1 and args.held % 2 == 0
    wp, fp, output = Path(args.weights), Path(args.features), Path(args.output)
    assert not output.exists()
    with np.load(wp) as archive:
        assert archive.files == ["weights"]
        weights = archive["weights"].astype(np.float64)
    with np.load(fp) as archive:
        # Explicitly do not read archive's prediction/reference arrays.
        features = archive["features"].astype(np.float64)
    depth, n, _ = weights.shape
    assert weights.shape == (depth, n, n) and features.shape == (depth, n, 13)
    root = Path(__file__).resolve().parents[2]
    source_manifest = json.loads((root / "results/phase2/ablation_v30_096_features_manifest.json").read_text())
    entries = json.loads((wp.parent / "inputs.json").read_text())
    match = [item for item in entries if Path(item["weights"]).resolve() == wp.resolve()]
    assert len(match) == 1
    identity = match[0]
    record = source_manifest["records"][identity["index"]]
    assert record["name"] == identity["name"] and record["seed"] == identity["seed"]
    assert hashlib.sha256(fp.read_bytes()).hexdigest() == record["sha256"]
    assert hashlib.sha256(wp.read_bytes()).hexdigest() == identity["sha256"]
    mu, gain = features[:, :, 1], features[:, :, 7]
    start = time.monotonic()
    a, sigma, means = build_model(weights, mu, gain)
    build_seconds = time.monotonic() - start
    costs64, costs32 = [scientific_cost(n, depth, args.train, dtype_rate=r) for r in (2, 1)]
    rows = []
    for seed in (261013711, 261013721):
        start = time.monotonic()
        train_y, train_g = sample_outputs(weights, mu, gain, a, means, args.train, seed)
        yc, gc = train_y - train_y.mean(axis=0), train_g - train_g.mean(axis=0)
        cov = np.mean(yc * gc, axis=0)
        varg = np.mean(gc * gc, axis=0)
        beta = np.divide(cov, varg, out=np.zeros_like(cov), where=varg > 1e-30)
        y, g = sample_outputs(weights, mu, gain, a, means, args.held, seed + 1)
        residual = y - beta * g
        variance = float(y.var(axis=0, ddof=1).mean())
        cvvar = float(residual.var(axis=0, ddof=1).mean())
        anti = antithetic_outputs(weights, args.held, seed + 2)
        row = {"train_seed": seed, "held_seed": seed + 1, "antithetic_seed": seed + 2,
               "train_samples": args.train, "held_samples": args.held,
               "raw_output_variance": variance, "cv_residual_variance": cvvar,
               "held_cv_variance_ratio": cvvar / variance,
               "antithetic_same_forward_variance_ratio": float(2 * anti.var(axis=0, ddof=1).mean() / variance),
               "beta_mean": float(beta.mean()), "beta_min": float(beta.min()), "beta_max": float(beta.max()),
               "held_control_mean_squared": float(np.mean(g.mean(axis=0)**2)),
               "elapsed_s": time.monotonic() - start, "projections": []}
        for cost in (costs64, costs32):
            count = cost["optimistic_cv_held_samples_after_pilot"]
            raw_proj = cvvar / count if count else None
            row["projections"].append({"dtype_billing_rate": cost["dtype_billing_rate"],
                                       "cv_budget_variance": raw_proj,
                                       "iid_budget_variance": variance / cost["optimistic_full_budget_iid_samples"],
                                       "projected_full_budget_adjusted": raw_proj,
                                       "full_budget_stage_gate_pass": raw_proj is not None and raw_proj < 5.779e-10})
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"version": "tenfold-multilevel-Gaussian-CV-r01", "complete": True, "checks": checks,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "weights_sha256": hashlib.sha256(wp.read_bytes()).hexdigest(),
              "features_sha256": hashlib.sha256(fp.read_bytes()).hexdigest(),
              "network_identity": {"index": identity["index"], "name": identity["name"], "seed": identity["seed"]},
              "feature_column_meanings": {"1": "frozenV29 preactivation mean mu", "7": "frozenV29 Gaussian expectedReLUgain Phi"},
              "feature_access": ["mu", "Phi"], "reference_access": False,
              "single_mechanism": "knownzero multilevel Gaussian-linear local innovations, inheritedthroughfrozenexpectedgates",
              "beta_rule": "peroutput ordinary train-only beta, two predeclared disjoint streams; no held refit",
              "build_seconds": build_seconds, "A_nbytes": int(a.nbytes),
              "sigma_range": [float(sigma.min()), float(sigma.max())],
              "costs_float64": costs64, "optimistic_float32_costs": costs32, "rows": rows,
              "predeclared_variance_gate": 5e-4, "actual_full_budget_gate": 5.779e-10,
              "variance_gate_pass": all(row["held_cv_variance_ratio"] < 5e-4 for row in rows),
              "adjusted_score": None,
              "limitations": ["Offline NumPy float64, no flopscope/resource/score pass",
                              "float32 cost projection is hypothetical; no float32 precision pass",
                              "First-order Gaussian skeleton and frozenfatherPhi need not explain deep residual",
                              "512 held samples are variance screen, not reference or precise confidence interval",
                              "High correlation with an approximate centered deeperactivation would not guarantee unbiasedness; this control's Gaussian Eg is exact by construction",
                              "3 dense forward chains including realnet, not a 2-forward cost; immutablecoefficients build also charged",
                              "Necessary variance gate only; matchedcost/no-bias/independentnetwork/allowed RNG required before candidate"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
