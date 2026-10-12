"""C20 analytic1024 cost gate consistent with measured component charge."""

import json
from pathlib import Path


def main():
    n = 1024
    r = 16
    # Both alternatives retain Xt/Yt GEMMs and cross product.
    common = 4 * n * n * r
    # Original linear: 3A Yt + Xt A w; final P product and row sum.
    original_linear = 7 * n * n - n
    # Regroup: AP, APw, two GEMMs, two thin products/sums and combine.
    regroup_linear = 2 * n * n + 4 * n * n * r + 4 * n * r + 2 * n
    p = Path("results/phase2/ablation_v30_020_exact_operation_cost.json")
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                n=n,
                rank=r,
                common_xt_yt_gemm_flops=common,
                original_linear_flops=original_linear,
                regroup_linear_flops=regroup_linear,
                linear_extra=regroup_linear - original_linear,
                conclusion="At productionn1024/r16 new2GEMMs overwhelm removed dense elementwise work. Copy/API/dtype charges excluded, actualcomponent already confirms increase; no valid cost-saving implementation yet.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
