"""C28 runtime alias poison oracle, stronger than lexical liveness only."""

import json
from pathlib import Path

import numpy as np


def main():
    rows = []
    for shape in [(7, 9), (16, 16), (3, 7, 9)]:
        owner = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        views = {
            "slice": owner[..., ::2],
            "transpose": owner.swapaxes(-1, -2),
            "reshape": owner.reshape(-1),
            "copy": owner.copy(),
        }
        saved = {name: view.copy() for name, view in views.items()}
        owner.fill(np.nan)
        for name, view in views.items():
            aliases = bool(np.shares_memory(owner, view))
            poisoned = bool(np.isnan(view).any())
            assert aliases == poisoned
            if not aliases:
                assert np.array_equal(view, saved[name])
            rows.append(dict(shape=shape, view=name, aliases=aliases, poison_detected=poisoned))
    out = Path("results/phase2/ablation_v30_028_poison_alias_contract.json")
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                rows=rows,
                conclusion="Runtime poison catches escaped slice/transpose/reshape views and spares copies. This proves oracle behavior, not live V29 pool reuse safety: remote-array aliases and actual use-after-recycle callsites still require instrumented integration.",
                scope="Offline alias diagnostic; no estimator modification or score",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
