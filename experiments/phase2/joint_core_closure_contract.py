"""H28 single object diag/offdiag consistency through actualtransport."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(58)
    rows = []
    for n in (4, 7, 11):
        g = rng.normal(size=n)
        cov = rng.normal(size=(n, n))
        cov = (cov + cov.T) / 2
        np.fill_diagonal(cov, 0)
        lam = 0.2
        core = np.diag(g) + lam * cov
        w = rng.normal(size=(n, n)) / np.sqrt(n)
        transported = w @ core @ w.T
        diagonal = (w * w) @ g + lam * np.diag(w @ cov @ w.T)
        off = transported - np.diag(diagonal)
        assert np.allclose(transported, np.diag(diagonal) + off, atol=1e-12)
        # Resetting offdiag to lambda* transportedcov discards offdiag of Wdiag(g)W.T.
        proxyoff = lam * (w @ cov @ w.T)
        np.fill_diagonal(proxyoff, 0)
        rows.append(
            dict(
                n=n,
                exact_reconstruction_error=float(
                    np.max(abs(transported - np.diag(diagonal) - off))
                ),
                reset_proxy_gap=float(np.linalg.norm(off - proxyoff)),
            )
        )
    path = Path("results/phase2/ablation_v30_058_joint_core_closure.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="One fixedcore object determines transporteddiag/offdiag jointly; independently resetting offdiag tolambdaC discardsdiagcore mixing. Selectednonlinearproduction and existingproxy de-duplication/candidatecost still pending.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
