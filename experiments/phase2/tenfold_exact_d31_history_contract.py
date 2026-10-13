"""Independent dense 31-support and two-step fullfeed history algebra contracts."""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from experiments.phase2.tenfold_exact_d31_history import (
    caps,
    feed_slices,
    nonlinear_feed,
    use_slices,
)

OUTPUT = Path("results/phase2/tenfold_exact_d31_history_contract.json")


def sym(*legs):
    subs = {3: "ir,jr,kr->ijk", 4: "ir,jr,kr,lr->ijkl"}[len(legs)]
    return sum(
        np.einsum(subs, *x, optimize=True) for x in itertools.permutations(legs)
    ) / math.factorial(len(legs))


def full_h(q, r):
    return 4 * sym(q, q, q, r)


def full_f(q, r, x, u):
    return sym(x, q, q) + sym(u, q, r)


def slices3(f):
    d21 = np.einsum("iij->ij", f).copy()
    np.fill_diagonal(d21, 0)
    return np.einsum("iii->i", f), d21


def tensor_feed(h, w1, w2):
    return 0.5 * (
        np.einsum("i,j,k,iijk->ijk", w2, w1, w1, h)
        + np.einsum("i,j,k,ijjk->ijk", w1, w2, w1, h)
        + np.einsum("i,j,k,ijkk->ijk", w1, w1, w2, h)
    )


def main():
    assert not OUTPUT.exists()
    rng = np.random.default_rng(31026)
    rows = []
    for n in (4, 8, 16):
        born31 = rng.normal(size=(n, n)) * 0.02
        np.fill_diagonal(born31, 0)
        q = np.eye(n)
        r = born31.T.copy()
        x = np.zeros((n, n))
        u = x.copy()
        wanted = np.zeros((n,) * 4)
        for i, j in itertools.permutations(range(n), 2):
            for p in set(itertools.permutations((i, i, i, j))):
                wanted[p] = born31[i, j]
        assert np.max(abs(full_h(q, r) - wanted)) < 1e-14
        h = full_h(q, r)
        f = full_f(q, r, x, u)
        steps = []
        for step in range(2):
            w = rng.normal(size=(n, n)) / math.sqrt(n)
            gain = rng.uniform(0.15, 0.85, n)
            w2 = rng.uniform(0.03, 0.4, n)
            z = rng.normal(size=n)
            targeth = np.einsum("ia,jb,kc,ld,abcd->ijkl", w, w, w, w, h, optimize=True)
            targetf = np.einsum("ia,jb,kc,abc->ijk", w, w, w, f, optimize=True)
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                qf, rf, xf, uf = [fnp.asarray(w) @ fnp.asarray(a) for a in (q, r, x, u)]
                actualuse = use_slices(qf, rf)
                actualprior = feed_slices(qf, rf, xf, uf)
                actualcaps = caps(qf, rf, fnp.asarray(z))
                qp, rp, newx, newu = nonlinear_feed(qf, rf, fnp.asarray(gain), fnp.asarray(w2))
                xp = xf * fnp.asarray(gain)[:, None] + newx
                up = uf * fnp.asarray(gain)[:, None] + newu
                actualpostfeed = feed_slices(qp, rp, xp, up)
            q, r, x, u = [np.asarray(a).copy() for a in (qp, rp, xp, up)]
            d31 = np.einsum("iiij->ij", targeth).copy()
            d22 = np.einsum("iijj->ij", targeth).copy()
            np.fill_diagonal(d31, 0)
            np.fill_diagonal(d22, 0)
            expecteduse = [np.einsum("iiii->i", targeth), d31, d22]
            feed = tensor_feed(targeth, gain, w2)
            priorpost = targetf * gain[:, None, None] * gain[None, :, None] * gain[None, None, :]
            expectedpost = priorpost + feed
            errors = {
                name: float(np.max(abs(np.asarray(a) - b)))
                for name, a, b in zip(("D4", "D31", "D22"), actualuse, expecteduse)
            }
            errors["H_fulltransport"] = float(
                np.max(
                    abs(
                        full_h(q, r)
                        - targeth
                        * gain[:, None, None, None]
                        * gain[None, :, None, None]
                        * gain[None, None, :, None]
                        * gain[None, None, None, :]
                    )
                )
            )
            errors["priorfeed_fulltensor"] = float(
                np.max(
                    abs(
                        full_f(np.asarray(qf), np.asarray(rf), np.asarray(xf), np.asarray(uf))
                        - targetf
                    )
                )
            )
            errors["newfeed_fulltensor"] = float(
                np.max(abs(full_f(q, r, np.asarray(newx), np.asarray(newu)) - feed))
            )
            errors["postfeed_fulltensor_once"] = float(
                np.max(abs(full_f(q, r, x, u) - expectedpost))
            )
            for name, a, b in zip(("priorD3", "priorD21"), actualprior, slices3(targetf)):
                errors[name] = float(np.max(abs(np.asarray(a) - b)))
            for name, a, b in zip(("postD3", "postD21"), actualpostfeed, slices3(expectedpost)):
                errors[name] = float(np.max(abs(np.asarray(a) - b)))
            fc = np.einsum("i,iijk->jk", z, targeth)
            pc = np.einsum("i,ijjk->jk", z, targeth)
            for j, k in itertools.product(range(n), repeat=2):
                if j == k:
                    fc[j, k] = pc[j, k] = 0
                else:
                    fc[j, k] -= z[j] * targeth[j, j, j, k] + z[k] * targeth[j, k, k, k]
                    pc[j, k] -= z[j] * targeth[j, j, j, k] + z[k] * targeth[j, j, k, k]
            for name, a, b in zip(("Fcap", "Pcap"), actualcaps, (fc, pc)):
                errors[name] = float(np.max(abs(np.asarray(a) - b)))
            assert max(errors.values()) < 1e-10, errors
            h = full_h(q, r)
            f = expectedpost
            steps.append(
                {
                    "step": step,
                    "errors": errors,
                    "legal_component_flops": ctx.summary_dict()["flops_used"],
                }
            )
        rows.append({"width": n, "steps": steps})
    n = 1024
    cube = n**3
    # Includes exact31 R/U transport, allUSE, and extra UQR priorfeed, excludescaps.
    incremental = (4 * 120 + 6 * 105 + 6 * 105) * cube
    report = {
        "version": "tenfold.exact31sharedhistory.contract",
        "complete": True,
        "rows": rows,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path("experiments/phase2/tenfold_exact_d31_history.py"))
        },
        "no_reference_or_target_access": True,
        "raw_mse": None,
        "adjusted_score": None,
        "costmodel1024depth16": {
            "newR_U_transport": 4 * 120 * cube,
            "newD31_D22_USE": 6 * 105 * cube,
            "prior_UQR_feed_readout": 6 * 105 * cube,
            "incremental_flops_excludescapsvectors": incremental,
            "existing_pureD4_strict_flops": 1902710926039,
            "naivecombined_flops": incremental + 1902710926039,
            "budget": 2**41,
            "passes_budget": False,
        },
        "scope": "Exact all31birth support, fulltensor transport/caps/feed; no history rank/age truncation; smallcontracts only",
        "limitations": [
            "newD22/211/1111birth stillomitted",
            "Naiveallhistory31 addedtopureD4 exceedsbudget, notcandidate",
            "No nonlinearproducer birthD31 implemented here",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
