"""Target-free pilot for random-plane angular averaging of homogeneous ReLU MLPs.

Research-only NumPy: no benchmark reference is loaded; no submitted score.
Each randomized quadrature node is marginally uniform on the unit sphere.
Multiplying by E[chi_n] gives an unbiased Gaussian mean in exact arithmetic
for a bias-free, positively homogeneous network. Finite grids are unbiased
over the random orthonormal plane and random phase, not exact per plane.
"""

# ruff: noqa: E402 -- freeze threads before importing NumPy
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


def radial_mean(n):
    return math.exp(0.5 * math.log(2.0) + math.lgamma((n + 1) / 2) - math.lgamma(n / 2))


def forward(weights, inputs):
    h = inputs
    for w in weights:
        # Estimator uses column W=w.T, hence the corresponding row batch is h@w.
        h = np.maximum(h @ w, 0)
    return h.astype(np.float64)


def planes(rng, groups, nodes, n):
    """Gaussian Gram-Schmidt makes a rotation-invariant orthonormal two-frame."""
    q = rng.standard_normal((groups, 2, n))
    q[:, 0] /= np.linalg.norm(q[:, 0], axis=1)[:, None]
    q[:, 1] -= np.sum(q[:, 1] * q[:, 0], axis=1)[:, None] * q[:, 0]
    q[:, 1] /= np.linalg.norm(q[:, 1], axis=1)[:, None]
    phases = rng.uniform(0, 2 * math.pi, groups)
    theta = phases[:, None] + 2 * math.pi * np.arange(nodes)[None, :] / nodes
    u = np.cos(theta)[:, :, None] * q[:, None, 0] + np.sin(theta)[:, :, None] * q[:, None, 1]
    return u.reshape(groups * nodes, n).astype(np.float32)


def means(weights, inputs, groups, nodes):
    # Keep memory bounded without changing the forward evaluations.
    ys = [forward(weights, inputs[i:i + 256]) for i in range(0, len(inputs), 256)]
    return np.concatenate(ys).reshape(groups, nodes, -1).mean(axis=1)


def pilot(weights, groups, nodes, seed):
    rng = np.random.default_rng(seed)
    n = weights.shape[1]
    count = groups * nodes
    er = radial_mean(n)
    start = time.monotonic()
    circle = er * means(weights, planes(rng, groups, nodes, n), groups, nodes)
    iid_x = rng.standard_normal((count, n), dtype=np.float32)
    iid = means(weights, iid_x, groups, nodes)
    sphere_x = rng.standard_normal((count, n))
    sphere_x /= np.linalg.norm(sphere_x, axis=1)[:, None]
    sphere = er * means(weights, sphere_x.astype(np.float32), groups, nodes)
    anti_x = rng.standard_normal((count // 2, n), dtype=np.float32)
    anti = means(weights, np.stack((anti_x, -anti_x), axis=1).reshape(count, n), groups, nodes)
    branches = {"iid": iid, "sphere_radial_rb": sphere, "sphere_plane": circle, "antithetic": anti}
    variances = {k: float(v.var(axis=0, ddof=1).mean()) for k, v in branches.items()}
    # All branches use groups*nodes forwards. This is a variance pilot only.
    result = {k: {"variance_of_group_mean": variances[k],
                  "equal_forward_variance_ratio_to_iid": variances[k] / variances["iid"],
                  "per_output_group_variance": v.var(axis=0, ddof=1).tolist()}
              for k, v in branches.items()}
    full_count = (2**41) // (2 * len(weights) * n**2 + len(weights) * n)
    for row in result.values():
        row["optimistic_full_budget_raw_variance"] = row["variance_of_group_mean"] * nodes / full_count
        row["optimistic_full_budget_adjusted_variance"] = row["optimistic_full_budget_raw_variance"]
    return {"seed": seed, "groups": groups, "nodes": nodes, "forwards_per_branch": count,
            "elapsed_s": time.monotonic() - start, "chi_mean": er,
            "branches": result, "optimistic_full_budget_forwards": int(full_count)}


def self_check():
    rng = np.random.default_rng(20261013)
    q = planes(rng, 4, 16, 8).reshape(4, 16, 8).astype(np.float64)
    assert np.max(abs(np.linalg.norm(q, axis=2) - 1)) < 1e-7
    w = rng.standard_normal((2, 8, 8))
    x = rng.standard_normal((8, 8))
    np.testing.assert_allclose(forward(w, 2 * x), 2 * forward(w, x), rtol=1e-12, atol=1e-12)
    asymmetric = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    np.testing.assert_array_equal(forward(asymmetric, np.array([[1.0, 0.0]])), [[1.0, 2.0]])
    # First-layer exact Gaussian mean is the contract, not competition reference.
    w1 = np.eye(2)[None, :, :]
    u = planes(rng, 1, 16384, 2)
    estimate = radial_mean(2) * forward(w1, u).mean(axis=0)
    np.testing.assert_allclose(estimate, np.full(2, 1 / math.sqrt(2 * math.pi)), atol=1e-7)
    return {"sphere_unit_norm": True, "positive_homogeneity": True, "api_weight_orientation": True,
            "two_dimensional_single_relu_contract": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights")
    parser.add_argument("--output")
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    checks = self_check()
    if args.self_check_only:
        print(json.dumps(checks))
        return
    assert args.weights and args.output and args.groups > 1 and args.nodes > 1 and args.nodes % 2 == 0
    path, output = Path(args.weights), Path(args.output)
    assert not output.exists()
    with np.load(path) as source:
        assert set(source.files) == {"weights"}, "Pilot accepts a target-free weights-only file."
        weights = source["weights"]
    assert weights.ndim == 3 and weights.shape[1] == weights.shape[2]
    rows = [pilot(weights, args.groups, args.nodes, seed) for seed in (261013501, 261013502)]
    report = {"version": "tenfold-angular-plane-r01", "complete": True,
              "weights": str(path.resolve()), "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "single_primary_mechanism": "random-plane circle averaging; analytic Gaussian radial mean",
              "checks": checks, "rows": rows,
              "predeclared_gate": {"target_adjusted": 5.779e-10,
                                    "required_standalone_full_budget_adjusted_variance": 5.779e-10,
                                    "do_not_promote_from_only_relative_sampling_gain": True},
              "scope": "Target-free variance pilot; no reference, score, or resource acceptance",
              "limitations": ["Cost projection omits plane construction, all movement, and RPC overhead",
                              "64 groups per seed yields noisy variance estimates; no confidence guarantee",
                              "Sphere and plane finite-precision normalization can create small bias",
                              "Network must be bias-free and positively homogeneous",
                              "Each branch uses an independent deterministic RNG stream; group variances do not prove benchmark improvement",
                              "Production implementation must use allowed flopscope operators"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "branches"} |
                      {"branches": {k: {kk: vv for kk, vv in v.items() if kk != "per_output_group_variance"}
                                    for k, v in row["branches"].items()}} for row in rows], indent=2))


if __name__ == "__main__":
    main()
