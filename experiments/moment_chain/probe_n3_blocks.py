#!/usr/bin/env python3
"""P2-1: measure (3)/(2,1)/(1,1,1) split of next-layer pre κ3 from atlas.

Uses keenanpepper bake_moments.third_order_blocks logic. Reports how large the
omitted n³ (b111) block is vs full κ3 and vs the storable (3)+(2,1) parts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dagger_atlas import load_shard_weights  # noqa: E402

ATLAS = "keenanpepper/arc-whestbench-higher-moments-2026"
OUT = Path(__file__).resolve().parent / "artifacts" / "n3_blocks_summary.json"


def connected(npz, prefix: str = ""):
    g = lambda k: np.asarray(npz[prefix + k], dtype=np.float64)
    mu, m2, m3 = g("mean"), g("m2"), g("m3")
    M11, M21, M22, M31 = g("M11"), g("M21"), g("M22"), g("M31")
    M12 = M21.transpose(0, 2, 1)
    mi, mj = mu[:, :, None], mu[:, None, :]
    m2i, m2j, m3i = m2[:, :, None], m2[:, None, :], m3[:, :, None]
    cov = M11 - mi * mj
    var = np.diagonal(cov, axis1=1, axis2=2)
    vi, vj = var[:, :, None], var[:, None, :]
    c21 = M21 - 2 * mi * M11 - m2i * mj + 2 * mi * mi * mj
    cen22 = (
        M22
        - 2 * mj * M21
        - 2 * mi * M12
        + mj * mj * m2i
        + mi * mi * m2j
        + 4 * mi * mj * M11
        - 3 * mi * mi * mj * mj
    )
    c22 = cen22 - vi * vj - 2 * cov * cov
    cen31 = (
        M31
        - mj * m3i
        - 3 * mi * M21
        + 3 * mi * mj * m2i
        + 3 * mi * mi * M11
        - 3 * mi**3 * mj
    )
    c31 = cen31 - 3 * vi * cov
    return dict(cov=cov, c21=c21, c22=c22, c31=c31)


def third_order_blocks(npz, weights: np.ndarray) -> dict[str, np.ndarray]:
    g = lambda k: np.asarray(npz[k], dtype=np.float64)
    Wt = np.asarray(weights, dtype=np.float64)
    post = connected(npz, prefix="")
    c21_all = post["c21"]
    pm1, pm2, pm3 = g("mean"), g("m2"), g("m3")
    mu3_post = pm3 - 3 * pm1 * pm2 + 2 * pm1**3
    qm1, qm2, qm3 = g("pre_mean"), g("pre_m2"), g("pre_m3")
    kappa3_pre = qm3 - 3 * qm1 * qm2 + 2 * qm1**3
    D = mu3_post.shape[0]
    full, b3, b21, b111 = [], [], [], []
    for ell in range(D - 1):
        W = Wt[ell + 1]
        c21 = c21_all[ell]
        mu3 = mu3_post[ell]
        blk3 = (W**3 * mu3[:, None]).sum(0)
        CW = c21 @ W
        unrestricted = ((W**2) * CW).sum(0)
        diag_overlap = (W**3 * np.diagonal(c21)[:, None]).sum(0)
        blk21 = 3.0 * (unrestricted - diag_overlap)
        f = kappa3_pre[ell + 1]
        full.append(f)
        b3.append(blk3)
        b21.append(blk21)
        b111.append(f - blk3 - blk21)
    return {
        "full": np.array(full),
        "b3": np.array(b3),
        "b21": np.array(b21),
        "b111": np.array(b111),
    }


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    weights_list, ids = load_shard_weights(n)
    rows = []
    for i, mid in enumerate(ids):
        path = hf_hub_download(
            ATLAS, f"full/mlp_{mid:05d}.npz", repo_type="dataset", local_files_only=True
        )
        z = dict(np.load(path))
        W = weights_list[i]  # (32,256,256)
        blocks = third_order_blocks(z, W)
        # identity check: full should match pre κ3 at layers 1..31
        for name in ("full", "b3", "b21", "b111"):
            assert blocks[name].shape[0] == 31
        f, b3, b21, b111 = blocks["full"], blocks["b3"], blocks["b21"], blocks["b111"]
        # energy shares by layer (mean over neurons of squares)
        e_f = np.mean(f**2, axis=1)
        e3 = np.mean(b3**2, axis=1)
        e21 = np.mean(b21**2, axis=1)
        e111 = np.mean(b111**2, axis=1)
        # relative residual of reconstruction
        recon = b3 + b21 + b111
        rel = float(np.sqrt(np.mean((recon - f) ** 2)) / (rms(f) + 1e-30))
        row = {
            "mlp_id": mid,
            "rms_full": rms(f),
            "rms_b3": rms(b3),
            "rms_b21": rms(b21),
            "rms_b111": rms(b111),
            "share_b3": float(np.mean(e3 / (e_f + 1e-30))),
            "share_b21": float(np.mean(e21 / (e_f + 1e-30))),
            "share_b111": float(np.mean(e111 / (e_f + 1e-30))),
            "recon_rel_err": rel,
            "depth_share_b111": e111.tolist(),
        }
        rows.append(row)
        print(
            f"mlp {mid:05d}: rms full={row['rms_full']:.3e} "
            f"shares b3/b21/b111={row['share_b3']:.3f}/{row['share_b21']:.3f}/{row['share_b111']:.3f} "
            f"recon_rel={rel:.2e}",
            flush=True,
        )

    summary = {
        "n": n,
        "mean_share_b3": float(np.mean([r["share_b3"] for r in rows])),
        "mean_share_b21": float(np.mean([r["share_b21"] for r in rows])),
        "mean_share_b111": float(np.mean([r["share_b111"] for r in rows])),
        "mean_rms_b111_over_full": float(
            np.mean([r["rms_b111"] / (r["rms_full"] + 1e-30) for r in rows])
        ),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
