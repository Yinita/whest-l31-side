"""H24 mapping: current lawful D3 proxy reweights old-source leg Gram."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(542)
    rows = []
    for n, rank in [(7, 2), (11, 3), (19, 4)]:
        a, p = rng.normal(size=(2, n, n))
        da, dp = rng.uniform(0.1, 1, size=(2, n))
        gram = (a * da) @ a.T + (p * dp) @ p.T
        # A final-mean proxy coefficient on D3 is phi/(6 sigma²).
        mu = rng.normal(size=n)
        sigma = rng.uniform(0.3, 2, size=n)
        sensitivity = np.exp(-0.5 * (mu / sigma) ** 2) / (np.sqrt(2 * np.pi) * 6 * sigma**2)
        # Normalize to preserve scale. This is a proxy, not full adjoint.
        scale = np.sqrt(sensitivity / sensitivity.mean())
        metric = scale[:, None] * gram * scale[None, :]
        omega = rng.normal(size=(n, rank))
        action = scale[:, None] * (
            (a * da) @ (a.T @ (scale[:, None] * omega))
            + (p * dp) @ (p.T @ (scale[:, None] * omega))
        )
        error = float(np.max(abs(action - metric @ omega)))
        assert error < 1e-10
        assert np.linalg.eigvalsh(metric).min() > -1e-10
        # Projection in scaled coordinates is oblique after mapping back;
        # QR after unscaling creates an orthogonal physical basis.
        scaled_basis, _ = np.linalg.qr(action)
        q, _ = np.linalg.qr(scaled_basis / scale[:, None])
        assert np.allclose(q.T @ q, np.eye(rank))
        rows.append(
            dict(
                n=n,
                rank=rank,
                operator_error=error,
                unscaled_orthogonality_error=float(np.max(abs(q.T @ q - np.eye(rank)))),
            )
        )
    path = Path("results/phase2/ablation_v30_054_output_metric_mapping.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                rule="D G D; D=sqrt(normalized legal current phi/(6 sigma²)); apply matrix-free, unscale then QR before physical leg projection",
                conclusion="Lawful proxy metric mapping checked. Next-layer coefficient requires approximate next moments; current sensitivity alone does not satisfy H24 next-layer specification. Full covariance feedback adjoint absent; production not accepted.",
                scope="Synthetic contract only, no before/after score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
