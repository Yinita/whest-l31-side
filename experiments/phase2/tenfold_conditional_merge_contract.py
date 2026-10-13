"""Exact target-free GH2-mixture moments and parent CP merge-source contract.

Small finite non-Gaussian conditional distributions validate all cumulants.
No candidate, SDK, model reference or shared research state is modified.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


def legal_merged_mean_cov(mean_plus, mean_minus, cov_plus, cov_minus):
    mean = (mean_plus+mean_minus)/2
    shift = (mean_plus-mean_minus)/2
    covariance = (cov_plus+cov_minus)/2+shift[:, None]*shift[None, :]
    return mean, covariance, shift


def legal_merge_source_factors(cov_plus, cov_minus, shift):
    """3 Sym(shift tensor Δcov), encoded by parent's rank-one feed CP block.

    A=Δcov, P=I, c1=3, c2=0, y=shift; all B1/M terms zero.
    This returns only a merged CP block, not arbitrary adapted parent stack.
    """
    delta = (cov_plus-cov_minus)/2
    width = len(shift)
    a = 3*delta
    b = fnp.eye(width, dtype=fnp.float64)
    c = shift[:, None]*fnp.ones((1, width), dtype=fnp.float64)
    return a, b, c


def distribution_moments(samples, probabilities):
    mean = np.sum(samples*probabilities[:, None], axis=0)
    centered = samples-mean
    covariance = np.einsum("s,si,sj->ij", probabilities, centered, centered)
    third = np.einsum("s,si,sj,sk->ijk", probabilities, centered, centered, centered)
    fourth = np.einsum("s,si,sj,sk,sl->ijkl", probabilities, centered, centered, centered, centered)
    covariance_pairs = np.einsum("ij,kl->ijkl", covariance, covariance)
    covariance_pairs += np.einsum("ik,jl->ijkl", covariance, covariance)
    covariance_pairs += np.einsum("il,jk->ijkl", covariance, covariance)
    return mean, covariance, third, fourth-covariance_pairs


def sym3(tensor):
    return sum(tensor.transpose(perm) for perm in itertools.permutations(range(3)))/6


def sym4(tensor):
    return sum(tensor.transpose(perm) for perm in itertools.permutations(range(4)))/24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    seed = 674901
    rng = np.random.default_rng(seed)
    rows = []
    for width in (4, 8):
        plus = rng.normal(size=(53, width))+rng.normal(size=width)
        minus = rng.normal(size=(61, width))*(rng.uniform(.5, 2, size=width))+rng.normal(size=width)
        pplus, pminus = rng.uniform(size=53), rng.uniform(size=61)
        pplus, pminus = pplus/pplus.sum(), pminus/pminus.sum()
        mp, cp, tp, kp = distribution_moments(plus, pplus)
        mm, cm, tm, km = distribution_moments(minus, pminus)
        expected = distribution_moments(np.concatenate((plus, minus)), np.concatenate((pplus/2, pminus/2)))
        d, dc, dt = (mp-mm)/2, (cp-cm)/2, (tp-tm)/2
        k3 = (tp+tm)/2+3*sym3(np.einsum("i,jk->ijk", d, dc))
        # Law of total cumulance for equal two-node mixtures. All coefficients
        # account for fully averaged Sym, not unnormalized permutation sums.
        k4 = (kp+km)/2+4*sym4(np.einsum("i,jkl->ijkl", d, dt))
        k4 += 3*sym4(np.einsum("ij,kl->ijkl", dc, dc))
        k4 -= 2*np.einsum("i,j,k,l->ijkl", d, d, d, d)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            fm, fc, fd = legal_merged_mean_cov(*map(fnp.asarray, (mp, mm, cp, cm)))
            af, bf, cf = legal_merge_source_factors(fnp.asarray(cp), fnp.asarray(cm), fd)
        raw = np.einsum("ir,jr,kr->ijk", np.asarray(af), np.asarray(bf), np.asarray(cf))
        cp_tensor = sym3(raw)
        cp_expected = 3*sym3(np.einsum("i,jk->ijk", d, dc))
        errors = {"mean_error": float(np.max(np.abs(np.asarray(fm)-expected[0]))),
                  "covariance_error": float(np.max(np.abs(np.asarray(fc)-expected[1]))),
                  "third_cumulant_error": float(np.max(np.abs(k3-expected[2]))),
                  "fourth_cumulant_error": float(np.max(np.abs(k4-expected[3]))),
                  "parent_feedCP_merge_source_error": float(np.max(np.abs(cp_tensor-cp_expected)))}
        assert max(errors.values()) < 1e-11, errors
        rows.append({"width": width, "conditional_supports": [53, 61], **errors,
                     "omitted_naive_K3_component_norm": float(np.linalg.norm(cp_expected)),
                     "omitted_naive_K4_component_norm": float(np.linalg.norm(k4-(kp+km)/2)),
                     "legal_mean_cov_CP_cost": context.summary_dict()})
    report = {"version": "tenfold.conditional.merge.r01", "complete": True, "seed": seed,
              "rows": rows, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "formulas": {"d": "(mu_plus-mu_minus)/2", "deltaC": "(C_plus-C_minus)/2",
                           "deltaK3": "(K3_plus-K3_minus)/2", "covariance": "Cbar+d d.T",
                           "K3": "K3bar+3Sym(d tensor deltaC)",
                           "K4": "K4bar+4Sym(d tensor deltaK3)+3Sym(deltaC tensor deltaC)-2 d^4"},
              "parent_merge_source_mapping": "A=deltaC,P=I,c1=3,c2=0,y=d, w2/e/s/feedback/ZL=0; rankn feed CP exactly 3Sym(d tensor deltaC)",
              "scope": "Exact equalweighttwo-distribution moment algebra and extra parentCP block only; no fullstack migration/generalfeed integration",
              "limitations": ["Conditional parent K3/K4 remain approximations", "Existingsource tensors must be weighted/copied, not averagedlegwise", "Parent sourcecapacity, agegates, sharedbasis and wrappedpermutation states need implementation changes", "K4proxy merge must retain diagnostic approximation: equalmean/cov average does not determine fullK4"],
              "no_adjusted_score": True, "not_resource_acceptance": True}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({"output": str(args.output), "rows": [{k: v for k, v in row.items() if k != "legal_mean_cov_CP_cost"} for row in rows]}, flush=True)


if __name__ == "__main__":
    main()
