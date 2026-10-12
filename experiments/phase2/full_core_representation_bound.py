"""O05 offline representation lower bound from full arbitrary fourth cumulant."""

import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(67)
    rows = []
    for n in (4, 7, 11):
        samples = rng.normal(size=(113, n)) ** 2
        z = samples - samples.mean(axis=0)
        covariance = z.T @ z / len(z)
        moment = np.einsum("si,sj,sk,sl->ijkl", z, z, z, z) / len(z)
        tensor = moment - (
            np.einsum("ij,kl->ijkl", covariance, covariance)
            + np.einsum("ik,jl->ijkl", covariance, covariance)
            + np.einsum("il,jk->ijkl", covariance, covariance)
        )
        trace = np.einsum("ijkk->ij", tensor)
        core = 6 * trace / (n + 4) - 3 * np.trace(trace) * np.eye(n) / ((n + 2) * (n + 4))
        diagonal = np.diag(np.diag(core))
        off = core - diagonal
        covoff = covariance - np.diag(np.diag(covariance))
        coef = float(np.sum(off * covoff) / np.sum(covoff * covoff))
        projection = diagonal + coef * covoff
        residual = core - projection
        singular = np.linalg.svd(residual, compute_uv=False)
        rows.append(
            dict(
                n=n,
                core_norm=float(np.linalg.norm(core)),
                diag_only_relative_error=float(np.linalg.norm(off) / np.linalg.norm(core)),
                diag_plus_best_lambda_relative_error=float(
                    np.linalg.norm(residual) / np.linalg.norm(core)
                ),
                oracle_lambda=coef,
                residual_singular_values=singular.tolist(),
            )
        )
    path = Path("results/phase2/ablation_v30_067_core_representation_bound.json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                rows=rows,
                scope="Synthetic finite-support full fourth cumulant, best scalar fit is oracle diagnostic only",
                conclusion="Diagonal+lambdaC approximation lacks general exact representation; lowrank residual spectrum recorded. Does not evaluate official truncated production on benchmark networks; official fullproduction/oracle candidate gate remains pending.",
                adjusted_score=None,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
