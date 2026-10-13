"""Joint proxy and signed diagonal residual projection/feed bookkeeping.

Independent signed dense tensor oracle, matrix-only legal E carrier. No truth.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.tenfold_sparse_d4_history import DiagonalHistory

OUTPUT = Path("results/phase2/tenfold_joint_harmonic_diagonal_contract_r34.json")


def pair(a, b):
    return (
        np.einsum("ij,kl->ijkl", a, b)
        + np.einsum("ik,jl->ijkl", a, b)
        + np.einsum("il,jk->ijkl", a, b)
        + np.einsum("ij,kl->ijkl", b, a)
        + np.einsum("ik,jl->ijkl", b, a)
        + np.einsum("il,jk->ijkl", b, a)
    ) / 6


def cp4(q, c):
    return np.einsum("ir,jr,kr,lr,r->ijkl", q, q, q, q, c, optimize=True)


def cp3(q, x):
    t = np.einsum("ir,jr,kr->ijk", x, q, q, optimize=True)
    return sum(t.transpose(p) for p in itertools.permutations(range(3))) / 6


def feed(h, gain, w2):
    return 0.5 * (
        np.einsum("i,j,k,iijk->ijk", w2, gain, gain, h)
        + np.einsum("i,j,k,ijjk->ijk", gain, w2, gain, h)
        + np.einsum("i,j,k,ijkk->ijk", gain, gain, w2, h)
    )


def d3_d21(t):
    d21 = np.einsum("iij->ij", t).copy()
    np.fill_diagonal(d21, 0)
    return np.einsum("iii->i", t), d21


def main():
    assert not OUTPUT.exists()
    rng = np.random.default_rng(34263)
    rows = []
    for n in (4, 8, 16):
        q = np.eye(n)
        coeff = rng.normal(scale=0.1, size=n)
        x = np.zeros((n, n))
        steps = []
        for step in range(2):
            raw = rng.normal(scale=0.02, size=(n, n))
            g = (raw + raw.T) / 2
            metric = 1.7
            epre = cp4(q, coeff)
            proxy = pair(g, metric * np.eye(n))
            h = proxy + epre
            gain = rng.uniform(0.15, 0.85, n)
            w2 = rng.uniform(0.05, 0.4, n)
            target4 = rng.normal(scale=0.03, size=n)
            # The parent proxy after this nonlinearity is a fresh model projection,
            # not silently an exact transport of the preproxy. Eold does transport.
            rawpost = rng.normal(scale=0.03, size=(n, n))
            gpost = (rawpost + rawpost.T) / 2
            postproxy = pair(gpost, metric * np.eye(n))
            fprior = cp3(q, x)
            newfeed = feed(epre, gain, w2)
            proxyfeed = feed(proxy, gain, w2)
            totalcurrentfeed = feed(h, gain, w2)
            assert np.max(abs(newfeed + proxyfeed - totalcurrentfeed)) < 1e-13
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                qf, cf, xf, w1f, w2f, t4f, pd4f = map(
                    fnp.asarray, (q, coeff, x, gain, w2, target4, metric * np.diag(gpost))
                )
                hist = DiagonalHistory(qf, cf)
                d4, d31, d22 = hist.slices()
                qpost = qf * w1f[:, None]
                inherit = fnp.sum((qpost * qpost) ** 2 * cf, axis=1)
                cnew = t4f - pd4f - inherit
                xnew = 1.5 * (w2f[:, None] * (qf * qf)) * cf[None, :]
                xpost = xf * w1f[:, None] + xnew
                fullq = fnp.concatenate([qpost, fnp.eye(n, dtype=fnp.float64)], axis=1)
                fullc = fnp.concatenate([cf, cnew])
                fullx = fnp.concatenate([xpost, fnp.zeros((n, n), dtype=fnp.float64)], axis=1)
                extraD3 = 1.5 * w2f * w1f * w1f * d4
                extraD21 = (
                    w2f[:, None] * w1f[:, None] * w1f[None, :] * d31
                    + 0.5 * w1f[:, None] ** 2 * w2f[None, :] * d22
                )
                # Zero repeated outputs like the parent diagonal split.
                extraD21 = extraD21 - fnp.diag(fnp.diag(extraD21))
            fullq, fullc, fullx = map(np.asarray, (fullq, fullc, fullx))
            representedpost = postproxy + cp4(fullq, fullc)
            expectedEpost = (
                epre
                * gain[:, None, None, None]
                * gain[None, :, None, None]
                * gain[None, None, :, None]
                * gain[None, None, None, :]
            )
            for i in range(n):
                expectedEpost[i, i, i, i] += (
                    target4[i] - metric * gpost[i, i] - np.sum(coeff * (q[i] * gain[i]) ** 4)
                )
            wantedpostfeed = (
                fprior * gain[:, None, None] * gain[None, :, None] * gain[None, None, :] + newfeed
            )
            errors = {
                "jointpost_D4_exact": float(
                    np.max(abs(np.einsum("iiii->i", representedpost) - target4))
                ),
                "jointpost_full_tensor": float(
                    np.max(abs(representedpost - postproxy - expectedEpost))
                ),
                "signedEfeed_fulltensor_once": float(
                    np.max(abs(cp3(fullq, fullx) - wantedpostfeed))
                ),
                "currentjoint_feed_linear_split": float(
                    np.max(abs(newfeed + proxyfeed - totalcurrentfeed))
                ),
            }
            for name, a, b in zip(
                ("newEfeedD3", "newEfeedD21"), (extraD3, extraD21), d3_d21(newfeed)
            ):
                errors[name] = float(np.max(abs(np.asarray(a) - b)))
            # Parent M residual must remove old total inherited K3 and both births.
            parentold = rng.normal(scale=0.01, size=(n,) * 3)
            parentold = sum(parentold.transpose(p) for p in itertools.permutations(range(3))) / 6
            oldtotal = parentold + fprior
            oldpost = oldtotal * gain[:, None, None] * gain[None, :, None] * gain[None, None, :]
            parentother = rng.normal(scale=0.01, size=(n,) * 3)
            parentother = (
                sum(parentother.transpose(p) for p in itertools.permutations(range(3))) / 6
            )
            allbirth = parentother + proxyfeed + newfeed
            wanted = rng.normal(scale=0.02, size=(n,) * 3)
            wanted = sum(wanted.transpose(p) for p in itertools.permutations(range(3))) / 6
            old3, old21 = d3_d21(oldpost)
            birth3, birth21 = d3_d21(allbirth)
            want3, want21 = d3_d21(wanted)
            s3, s21 = want3 - old3 - birth3, want21 - old21 - birth21
            errors["ordinaryM_D3_repeated_exact_dedup"] = float(
                np.max(abs(s3 + old3 + birth3 - want3))
            )
            errors["ordinaryM_D21_repeated_exact_dedup"] = float(
                np.max(abs(s21 + old21 + birth21 - want21))
            )
            # Two fixed broken ledgers certify the contract can catch duplication.
            duplicateD4 = float(
                np.max(
                    abs(
                        np.einsum(
                            "iiii->i",
                            postproxy
                            + cp4(
                                fullq,
                                np.concatenate(
                                    [
                                        coeff,
                                        target4 - np.sum((q * gain[:, None]) ** 4 * coeff, axis=1),
                                    ]
                                ),
                            ),
                        )
                        - target4
                    )
                )
            )
            doublefeed = float(
                np.max(abs(d3_d21(wantedpostfeed + newfeed)[0] - d3_d21(wantedpostfeed)[0]))
            )
            assert duplicateD4 > 1e-8 and doublefeed > 1e-8
            assert max(errors.values()) < 1e-12, errors
            q, coeff, x = fullq.copy(), fullc.copy(), fullx.copy()
            # Real W transports shared q/x and static signed coefs once.
            w = rng.normal(scale=1 / np.sqrt(n), size=(n, n))
            oldh = cp4(q, coeff)
            oldf = cp3(q, x)
            q, x = w @ q, w @ x
            errors["next_rawW_CP4"] = float(
                np.max(
                    abs(
                        cp4(q, coeff)
                        - np.einsum("ia,jb,kc,ld,abcd->ijkl", w, w, w, w, oldh, optimize=True)
                    )
                )
            )
            errors["next_rawW_CP3"] = float(
                np.max(
                    abs(cp3(q, x) - np.einsum("ia,jb,kc,abc->ijk", w, w, w, oldf, optimize=True))
                )
            )
            assert max(errors.values()) < 1e-12
            steps.append(
                {
                    "step": step,
                    "errors": errors,
                    "broken_proxyD4_doublecount": duplicateD4,
                    "broken_newfeed_doublecount": doublefeed,
                    "legal_joint_component_flops": ctx.summary_dict()["flops_used"],
                }
            )
        rows.append({"width": n, "steps": steps})
    report = {
        "version": "tenfold.jointHarmonicDiagonal.r34.contract",
        "complete": True,
        "rows": rows,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(
            Path("experiments/phase2/tenfold_sparse_d4_history.py").read_bytes()
        ).hexdigest(),
        "no_reference_or_target_access": True,
        "raw_mse": None,
        "adjusted_score": None,
        "scope": "Dense independent signed jointprojection/feed/dedup/rawW algebra; candidate actualintegration separate",
        "limitations": [
            "Parentharmoniccross support remainsapproximate",
            "No exactGaussian22production or1024resourceacceptance",
            "Arbitrarysignedtoy verifiesalgebra notdistributionaccuracy",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
