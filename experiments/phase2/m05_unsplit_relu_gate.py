"""Diagnostic: original WGMprop unsplit UT versus exact first-ReLU mean.

The exact mean is analytic, not a benchmark reference. This evaluates only the
unsplit UT boundary case and never claims the complete adaptive GMM method.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path, output = Path(args.weights), Path(args.output)
    assert not output.exists()
    weight = np.load(path)["weights"][0].astype(np.float64)
    n = weight.shape[0]
    analytic = np.sqrt(np.sum(weight**2, axis=0)) / math.sqrt(2 * math.pi)
    rows = []
    # Original implementation lambda=n-3 for n>=3; zero-lambda is a diagnostic control.
    for kappa in (n - 3, 0):
        # Points {0, +/-sqrt(n+kappa)e_i}; ReLU positive homogeneity
        # collapses the UT weighted first-layer mean to the column l1 norm.
        unsplit = np.sum(np.abs(weight), axis=0) / (2 * math.sqrt(n + kappa))
        rows.append({"kappa": kappa, "central_weight": kappa / (n + kappa),
                     "n_points": 2 * n + 1,
                     "raw_mse_vs_exact_first_layer": float(np.mean((unsplit - analytic)**2)),
                     "mean_relative_bias": float(np.mean(unsplit / analytic - 1)),
                     "per_neuron_exact_mean": analytic.tolist(),
                     "per_neuron_ut_mean": unsplit.tolist()})
    # Verify collapsed formula against explicit sigma-point evaluation on small
    # arbitrary Gaussian shapes. No network distribution approximation here.
    rng = np.random.default_rng(61013005)
    errors = []
    for width in (3, 5, 9):
        w = rng.normal(size=(width, width))
        kappa = width - 3
        points = np.vstack((np.zeros((1, width)), np.eye(width), -np.eye(width)))
        points *= math.sqrt(width + kappa)
        weights_ut = np.full(2 * width + 1, 1 / (2 * (width + kappa)))
        weights_ut[0] = kappa / (width + kappa)
        explicit = weights_ut @ np.maximum(points @ w, 0)
        collapsed = np.sum(abs(w), axis=0) / (2 * math.sqrt(width + kappa))
        error = float(abs(explicit - collapsed).max())
        assert error < 1e-12
        errors.append({"width": width, "max_abs_error": error})
    report = {"version": "M05-unsplit-UT-ReLU-r02", "complete": True,
              "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_commit": "915eaacea0ae3906536b8ffc6662a84b43b8487c",
              "source_function": "utils_wgmprop_propagation.py::_subspace_ut_sampling_from_level2",
              "rows": rows, "collapsed_formula_contract": errors,
              "scope": "Offline unsplit first-layer UT diagnostic, analytic truth; no benchmark score",
              "conclusion": "Original kappa=n-3 unsplit rule can badly bias ReLU means; adaptive splitting must be validated before use. This does not reject complete WGMprop.",
              "limitations": ["one existing research network", "first layer only",
                              "zero-kappa comparison is diagnostic, not a new full algorithm",
                              "no adaptive splitting/criterion/full-network port yet"]}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if not k.startswith('per_neuron')}
                      for r in rows], indent=2))


if __name__ == "__main__":
    main()
