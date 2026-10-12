"""Exercise actual V29 persistent pool across shapes, offline resource diagnostic."""

import importlib.util
import json
import resource
from pathlib import Path

import flopscope as flops


def main():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("v29", root / "estimators/k3_v29/estimator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pool = module._Pool(module.fnp.float32)
    rows = []
    with flops.BudgetContext(flop_budget=2**41):
        for n in [4, 8, 4, 16, 8]:
            buffer = pool.get("shape_probe", (n, n))
            rows.append(
                dict(
                    width=n,
                    allocated_keys=len(pool.bufs),
                    retained_bytes=sum(b.size * 4 for b in pool.bufs.values()),
                    shape=list(buffer.shape),
                    rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
            )
    output = root / "results/phase2/ablation_v30_026_pool_shape_probe.json"
    assert not output.exists()
    output.write_text(
        json.dumps(
            dict(
                scope="Actual pool class synthetic shape lifetime, not full1024 predict resource audit",
                rows=rows,
                conclusion="Old shapes retained; repeated shapes reused. Maximum live estimator scratch and full resource gate remain unmeasured",
            ),
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()
