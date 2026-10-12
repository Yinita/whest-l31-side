"""L14 constrained four heads: equivalent continuous hinge representation.

For all other features free at each alpha boundary, adjacent coefficients must
match except alpha and intercept. Three hinges exactly encode these constraints.
This evaluates the frozen constrained model offline, with no charged-score claim.
"""

import json
from pathlib import Path

import numpy as np
from learning_feature_screens import fit_predict


def main():
    root = Path(__file__).resolve().parents[2]
    data = [
        dict(np.load(root / f"benchmark/data/v29_features_v30_096/network_{i}.npz"))
        for i in range(8)
    ]
    rows = []
    boundary_errors = []
    for held in range(8):
        train = [i for i in range(8) if i != held]
        x = [d["features"][-1].astype(np.float64) for d in data]
        y = [(d["reference"][-1] - d["predictions"][-1]).astype(np.float64) for d in data]
        alpha = np.concatenate([x[i][:, 4] for i in train])
        counts = np.histogram(alpha, bins=[-np.inf, -1, 0, 1, np.inf])[0]
        knots = [k for j, k in enumerate((-1, 0, 1)) if min(counts[j], counts[j + 1]) >= 100]
        features = [
            np.concatenate([a, np.maximum(a[:, 4, None] - np.array(knots)[None, :], 0)], axis=1)
            for a in x
        ]
        delta = fit_predict([features[i] for i in train], [y[i] for i in train], features[held])
        # Independent two-sided boundary evaluation, all non-alpha inputs held fixed.
        for knot in knots:
            pair = np.repeat(x[held][:8], 2, axis=0)
            pair[:, 4] = np.tile([knot - 1e-9, knot + 1e-9], 8)
            pair = np.concatenate(
                [pair, np.maximum(pair[:, 4, None] - np.array(knots)[None, :], 0)], axis=1
            )
            p = fit_predict([features[i] for i in train], [y[i] for i in train], pair)
            boundary_errors.append(float(np.max(abs(p[::2] - p[1::2]))))
        before = float(np.mean(y[held] ** 2))
        after = float(np.mean((y[held] - delta) ** 2))
        rows.append(
            dict(
                network_id=held,
                before_raw=before,
                after_raw=after,
                raw_ratio=after / before,
                train_interval_counts=counts.tolist(),
                active_knots=knots,
            )
        )
    out = dict(
        scope="Development network LOO only; no independent confirmation",
        per_network=rows,
        before_raw=float(np.mean([r["before_raw"] for r in rows])),
        after_raw=float(np.mean([r["after_raw"] for r in rows])),
        max_boundary_jump_at_2e_9=max(boundary_errors),
        adjusted_score=None,
        conclusion="Constrained original four heads evaluated through exactly equivalent hinge basis. Other-feature coefficients shared, alpha/intercept piecewise with continuity. Charged candidate and parent CV gate remain unpassed.",
    )
    path = root / "results/phase2/ablation_v30_109_continuous_heads_cv.json"
    assert not path.exists()
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(out["after_raw"] / out["before_raw"], out["max_boundary_jump_at_2e_9"])


if __name__ == "__main__":
    main()
