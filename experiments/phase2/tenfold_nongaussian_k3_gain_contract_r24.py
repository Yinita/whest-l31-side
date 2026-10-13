"""Fixed Edgeworth derivative, legal K3 transport and centering contracts.

No whole1024predict, reference target, fitting, or sharedstateedit.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import SetupContext

from local_engine import build_mlp

SOURCE = Path("estimators/tenfold_nongaussian_k3_gain_r24/estimator.py")
OUTPUT = Path("results/phase2/tenfold_nongaussian_k3_gain_contract_r24.json")


def sym(a, b, c):
    return sum(np.einsum("it,jt,kt->ijk", *x) for x in itertools.permutations((a, b, c))) / 6


def main():
    assert not OUTPUT.exists()
    spec = importlib.util.spec_from_file_location("gain24contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(24114)
    # Independent formal Edgeworthdensity Hermiteintegration forindicatorX>0.
    nodes, weights = np.polynomial.legendre.leggauss(256)
    formula_rows = []
    for alpha, sigma, k3std, k4std in (
        (-2, 0.7, 0.04, 0.025),
        (-0.3, 1.1, 0.08, -0.04),
        (0, 1, 0.1, 0.1),
        (1.5, 0.9, -0.05, 0.08),
    ):
        lower = -alpha
        z = lower + (nodes + 1) * (12 - lower) / 2
        pdf = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
        h3 = z**3 - 3 * z
        h4 = z**4 - 6 * z * z + 3
        quadrature = (
            np.sum(weights * pdf * (1 + k3std * h3 / 6 + k4std * h4 / 24)) * (12 - lower) / 2
        )
        phi = math.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi)
        cdf = (1 + math.erf(alpha / math.sqrt(2))) / 2
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            gain = module.expected_relu_gain(
                *map(
                    fnp.asarray,
                    (
                        np.array([cdf]),
                        np.array([alpha]),
                        np.array([sigma]),
                        np.array([phi]),
                        np.array([k3std * sigma**3]),
                        np.array([k4std * sigma**4]),
                    ),
                )
            )
        error = abs(float(np.asarray(gain)[0]) - quadrature)
        assert error < 1e-12
        formula_rows.append(
            {
                "alpha": alpha,
                "sigma": sigma,
                "indicator_gain": float(np.asarray(gain)[0]),
                "Hermitedensity_quadrature_gain": float(quadrature),
                "error": error,
                "legal_flops": ctx.summary_dict()["flops_used"],
            }
        )
    tensor_rows = []
    for n in (4, 8, 16):
        a, p, h = [rng.normal(size=(n, 3)) for _ in range(3)]
        tensor = sym(a, p, h)
        gate = rng.uniform(0.1, 0.9, size=n)
        correction = rng.normal(scale=0.01, size=n)
        gain = gate + correction
        wanted = tensor * gain[:, None, None] * gain[None, :, None] * gain[None, None, :]
        represented = sym(a * gain[:, None], p * gain[:, None], h * gain[:, None])
        err = float(np.max(abs(wanted - represented)))
        assert err < 1e-12
        d3 = np.einsum("iii->i", tensor)
        d21 = np.array([[tensor[i, i, j] if i != j else 0 for j in range(n)] for i in range(n)])
        ds3 = d3 * gain**3
        ds21 = d21 * gain[:, None] ** 2 * gain[None, :]
        assert np.max(abs(ds3 - np.einsum("iii->i", wanted))) < 1e-12
        # A consistent newbornM subtracts newinherited gain slices exactly.
        pk3 = rng.normal(size=n)
        pk21 = rng.normal(size=(n, n))
        np.fill_diagonal(pk21, 0)
        new3, new21 = pk3 - ds3, pk21 - ds21
        assert np.max(abs(new3 + ds3 - pk3)) < 1e-12
        assert np.max(abs(new21 + ds21 - pk21)) < 1e-12
        # Partialgain changes (onlyreadout ortransport) produce realcounterexamples.
        wrong3 = pk3 - d3 * gate**3
        inconsistency = float(np.max(abs(wrong3 + ds3 - pk3)))
        assert inconsistency > 1e-8
        tensor_rows.append(
            {
                "width": n,
                "CPgain_transport_max_error": err,
                "wrong_birth_subtraction_counterexample": inconsistency,
            }
        )
    smoke = []
    for n in (32, 64):
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            mlp = build_mlp(n, 4, 2424 + n)
            estimator = module.Estimator()
            estimator.setup(
                SetupContext(width=n, depth=4, flop_budget=2**41, api_version="0.1", seed=0)
            )
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            got = estimator.predict(mlp, 2**41)
        assert np.isfinite(np.asarray(got)).all() and np.asarray(got).shape == (4, n)
        smoke.append(
            {
                "width": n,
                "depth": 4,
                "allfinite": True,
                "wholepredict_legal_flops": ctx.summary_dict()["flops_used"],
            }
        )
    # ActualcachedV29state; no reference or MC momentused.
    meta = json.loads(Path("results/phase2/ablation_v30_067_r24_real_k31_capture.json").read_text())
    path = Path(meta["capture"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["capture_sha256"]
    with np.load(path) as src:
        alpha, sigma, d3, g4 = [
            src[k].astype(np.float64) for k in ("alpha", "sigma", "D3", "g4row")
        ]
    phi = np.exp(-(alpha**2) / 2) / np.sqrt(2 * np.pi)
    gate = np.array([(1 + math.erf(a / math.sqrt(2))) / 2 for a in alpha])
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        corrected = module.expected_relu_gain(*map(fnp.asarray, (gate, alpha, sigma, phi, d3, g4)))
    gains = np.asarray(corrected)
    delta = gains - gate
    report = {
        "version": "tenfold.nongaussian.K3gain.r24",
        "complete": True,
        "candidate_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixed_theoretical_coefficients": {"kappa3_w4": 1 / 6, "kappa4_w5": 1 / 24},
        "formalHermitecontracts": formula_rows,
        "tensor_transport_and_centering_contracts": tensor_rows,
        "wholepredict_smoke": smoke,
        "actual_cached_network0_layer14_gain": {
            "capture_sha256": meta["capture_sha256"],
            "gaussian_gain_min": float(gate.min()),
            "gaussian_gain_max": float(gate.max()),
            "corrected_gain_min": float(gains.min()),
            "corrected_gain_max": float(gains.max()),
            "delta_rms": float(np.sqrt(np.mean(delta**2))),
            "delta_max_abs": float(abs(delta).max()),
            "delta_mean": float(delta.mean()),
            "negative_gain_count": int((gains < 0).sum()),
            "above_one_gain_count": int((gains > 1).sum()),
            "legal_component_flops": ctx.summary_dict()["flops_used"],
        },
        "no_reference_or_target_access": True,
        "raw_mse": None,
        "adjusted_score": None,
        "scope": "MarginalEdgeworthderivative foroldK3transport plusconsistentbirthsubtraction; no formalfullK3highergradeclosure orscore",
        "limitations": [
            "Gainproduct includeshighergradecrossproducts not officialbudget3terms",
            "Inputmarginalg4/D3 retainparentproxies",
            "No clipping/fittedscale; actualgainoutsiderange reported",
            "Full1024strictresources andscore pending",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
