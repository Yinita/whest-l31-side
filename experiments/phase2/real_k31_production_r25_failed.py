"""Full official augmented K31 polynomial on actual frozen V29 layer14 slices.

Input fourth-order slices still inherit V29's existing proxies. This is not a
replacement for the full fourth-order core and does not report a score.
"""

# ruff: noqa: E402
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from full_k31_charged_production import wick

from experiments.phase2.audit_official_augmented_terms import coefficient, extract_generators


def legal_wicks(mu, sigma):
    alpha = mu/sigma
    phi = flops.stats.norm.pdf(alpha).astype(fnp.float32)
    gate = flops.stats.norm.cdf(alpha).astype(fnp.float32)
    inv = 1/sigma
    result = {(1, 1): gate, (2, 1): phi*inv, (3, 1): -alpha*phi*inv**2,
              (4, 1): (alpha*alpha-1)*phi*inv**3,
              (5, 1): (3*alpha-alpha**3)*phi*inv**4,
              (1, 3): 3*((mu*mu+sigma*sigma)*gate+mu*sigma*phi),
              (2, 3): 6*(sigma*phi+mu*gate), (3, 3): 6*gate,
              (4, 3): 6*phi*inv, (5, 3): -6*alpha*phi*inv**2}
    return result


def factor_table(data):
    return {(1, 1): data["C_off"], (2, 1): data["D21"], (1, 2): data["D21"].T,
            (3, 0): data["D3"], (0, 3): data["D3"],
            (4, 0): data["g4row"], (0, 4): data["g4row"],
            (2, 2): data["wk4m"], (3, 1): data["wk431"], (1, 3): data["wk431"].T}


def main():
    output = Path("results/phase2/ablation_v30_067_r25_real_k31_production.json")
    assert not output.exists()
    mp = Path("results/phase2/ablation_v30_067_r24_real_k31_capture.json")
    meta = json.loads(mp.read_text())
    capture = Path(meta["capture"])
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == meta["capture_sha256"]
    with np.load(capture) as loaded:
        all_data = {k: loaded[k].copy() for k in loaded.files}
    # Reuse pinned local originals, never rely on a disposable /tmp clone.
    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)/"src/mlp_kprop"
        directory.mkdir(parents=True)
        for name in ("partitions.py", "kprop_harmonic.py"):
            shutil.copyfile(Path("benchmark/data/official_source_93d091a4")/name, directory/name)
        namespace, sources = extract_generators(Path(td))
        assert [s["sha256"] for s in sources] == [
            "acb944b9b6f5823affc9baae5946662330abda658ae79c08294ee32aec4536dd",
            "45e5fcc030bdb6bfba526898fdd53a650b83193e992cfad51e9422ccb70e11c1"]
        terms = namespace["get_all_terms_iso"](3, d_max=4, augment=True)[(3, 1)]
    rows = []
    for n in (8, 32, 1024):
        data = {k: val[:n, :n].copy() if val.ndim == 2 else val[:n].copy()
                for k, val in all_data.items()}
        fs = factor_table(data)
        pairs = {(sum(v[i] for v in part), pw) for part in terms for i, pw in enumerate((3, 1))}
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            factors = {v: fnp.asarray(x) for v, x in fs.items()}
            wicks = legal_wicks(fnp.asarray(data["mu"]), fnp.asarray(data["sigma"]))
            pk31 = fnp.zeros((n, n), dtype=fnp.float32)
            for part, count in terms.items():
                value = fnp.ones((n, n), dtype=fnp.float32)*float(coefficient(part, count))
                for v in part:
                    value = value*fnp.reshape(factors[v], tuple(n if s else 1 for s in v))
                value = value*wicks[(sum(v[0] for v in part), 3)][:, None]
                value = value*wicks[(sum(v[1] for v in part), 1)][None, :]
                pk31 = pk31+value
            fnp.fill_diagonal(pk31, 0)
            mean, moment2, pk21, cov = map(fnp.asarray, [data[k] for k in ("pk1v", "pk2v", "pk21", "pk11")])
            k31 = pk31-3*mean[:, None]*pk21+(6*mean*mean-3*moment2)[:, None]*cov
            fnp.fill_diagonal(k31, 0)
            core = (k31+k31.T)*(6/(n+4))
        # Independent scalar diagram sum and already independently verified
        # offline Gaussian Wick functions. No target/reference arrays used.
        offline_w = {pair: wick(*pair, data["mu"].astype(np.float64),
                                 data["sigma"].astype(np.float64)) for pair in pairs}
        wick_error = max(float(np.max(np.abs(np.asarray(wicks[p])-offline_w[p]))) for p in pairs)
        entries = [(i, j) for i in range(n) for j in range(n)] if n <= 32 else [
            (i*61, (i*67+5)%n) for i in range(16)]
        expected_raw, expected_centered = [], []
        for i, j in entries:
            value = 0.0
            if i != j:
                for part, count in terms.items():
                    term = float(coefficient(part, count))
                    for v in part:
                        term *= float(fs[v][tuple(x for x, support in zip((i, j), v) if support)])
                    term *= float(offline_w[(sum(v[0] for v in part), 3)][i])
                    term *= float(offline_w[(sum(v[1] for v in part), 1)][j])
                    value += term
            expected_raw.append(value)
            centered = value-3*float(data["pk1v"][i])*float(data["pk21"][i, j])
            centered += (6*float(data["pk1v"][i])**2-3*float(data["pk2v"][i]))*float(data["pk11"][i, j])
            expected_centered.append(centered if i != j else 0)
        raw_actual = np.array([np.asarray(pk31)[i, j] for i, j in entries])
        actual = np.array([np.asarray(k31)[i, j] for i, j in entries])
        raw_error = float(np.max(np.abs(raw_actual-expected_raw)))
        centered_error = float(np.max(np.abs(actual-expected_centered)))
        assert raw_error < 5e-5*max(float(np.max(np.abs(expected_raw))), 1e-12)+1e-8
        assert centered_error < 5e-5*max(float(np.max(np.abs(expected_centered))), 1e-12)+1e-8
        assert np.isfinite(np.asarray(core)).all()
        assert np.max(np.abs(np.asarray(core)-np.asarray(core).T)) == 0
        rows.append({"width": n, "diagram_classes": len(terms), "wick_max_abs_error": wick_error,
                     "fixed_entries": len(entries), "pK31_scalar_max_abs_error": raw_error,
                     "K31_scalar_max_abs_error": centered_error,
                     "pK31_norm": float(np.linalg.norm(np.asarray(pk31).astype(np.float64))),
                     "K31_norm": float(np.linalg.norm(np.asarray(k31).astype(np.float64))),
                     "projected_K31_core_norm": float(np.linalg.norm(np.asarray(core).astype(np.float64))),
                     "legal_cost": ctx.summary_dict()})
        print({k:v for k,v in rows[-1].items() if k!='legal_cost'}, flush=True)
    report = {"version": "v30.067.r25", "complete": True, "rows": rows,
              "before": "Synthetic K31 factors with offline supplied Wick table",
              "after": "Actual V29 layer14 input slices, legal p1/p3 Wick k1..5, all15 augmented K31 diagrams and centered conversion",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "capture_metadata_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
              "capture_sha256": meta["capture_sha256"], "official_sources": sources,
              "dataset_commit": meta["dataset_commit"], "network_id": meta["network_id"], "layer": 14,
              "no_adjusted_score": True, "not_resource_acceptance": True,
              "limitations": ["Actual fourth-order input slices still use V29 proxies, not oracle or true full fourth cumulants",
                              "Complete K211/CovCov and proxy subtraction integration still unresolved",
                              "Core component diagnostic is not complete predictor/score"]}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
