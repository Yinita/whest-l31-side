"""Guarded immutable template for six fixed pregraph teacher-state features.

Oracle truth enters this offline-only worker; no candidate or learned model score.
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

SOURCE = Path("estimators/sprint_public_v37b_h18_r01/estimator.py")
SOURCE_SHA = "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
DATA_REV = "cc3405a9393bc398e0af4b0e5d6e78da0ad5f6c8"
DATA_URL = f"https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k/resolve/{DATA_REV}/shard00.npz?download=true"
DATA_SHA = "aab6930a8a82b9dfe792a0e71456873165f73d143faac06fc32e5ca52b2b0fa5"
OUT = Path("results/phase2/modal_pregraph_teacher_capture_r50")
INDICES = list(range(0, 384, 16))
FEATURES = ["ones", "mu", "var", "sigma", "alpha", "abs_alpha", "phi", "Phi",
            "pk1", "D3", "D21norm", "K4_sigma", "sigma_phi",
            "skew_H2phi", "kurt_H3phi", "skew2_H5phi", "skewkurt_H6phi",
            "kurt2_H7phi", "skew", "kurt", "gate", "depth_fraction"]
app = modal.App("whest-pregraph-teacher-capture-r50")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.2.6", "flopscope==0.12.1", "whestbench==0.16.1")
         .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
         .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
         .add_local_file(str(SOURCE), "/code/h18.py")
         .add_local_file("experiments/phase2/tenfold_graph_features_r50.py", "/code/tenfold_graph_features_r50.py")
         .add_local_file("experiments/phase2/capture_pregraph_state_r50.py", "/code/capture_pregraph_state_r50.py"))


@app.function(image=image, cpu=(1, 1), memory=(16384, 16384), timeout=300,
              retries=0, max_containers=24, scaledown_window=2)
def rollout(job):
    import importlib.metadata
    import importlib.util
    import re
    import resource
    import sys
    from types import SimpleNamespace

    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np
    import torch
    from whestbench import SetupContext

    sys.path.insert(0,"/code")
    from capture_pregraph_state_r50 import GraphCapture

    idx, expected_weight_hash, truth_bytes = job
    truth_post = np.frombuffer(truth_bytes, dtype=np.float64).reshape(15, 1024, 4)
    started = time.monotonic()
    try:
        for key in tuple(os.environ):
            if re.match(r"V\d+_", key):
                os.environ.pop(key)
        source = Path("/code/h18.py")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
        spec = importlib.util.spec_from_file_location("h18_corpus_r38", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        flops.configure(symmetry_warnings=False)  # Offline warning suppression only.
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu").manual_seed(770000 + idx)
        weights = (torch.randn(16, 1024, 1024, generator=generator, dtype=torch.float32)
                   * (2.0 / 1024) ** 0.5).numpy()
        weights_sha = hashlib.sha256(weights.tobytes()).hexdigest()
        assert weights_sha == expected_weight_hash
        estimator = module.Estimator()
        with flops.BudgetContext(flop_budget=2**41, quiet=True):
            estimator.setup(SetupContext(width=1024, depth=16, flop_budget=2**41,
                                         api_version="0.1", seed=0))
        lines = source.read_text().splitlines()
        pretrigger = next(i for i, line in enumerate(lines, 1) if "if riders and not NO_CORR:" in line)
        posttrigger = next(i for i, line in enumerate(lines, 1) if "mu = K1v if delta is None else K1v + delta" in line)
        features, states, scales = [], [], []
        graph_capture=GraphCapture(1024,16)
        pre_seen, post_seen = set(), set()
        clipped_mean = clipped_var = 0
        after_states = []
        actual_after_states = []
        covariance_diagnostics = []

        def tracer(frame, event, arg):
            nonlocal clipped_mean, clipped_var
            if event != "line" or frame.f_code.co_filename != str(source):
                return tracer
            state = frame.f_locals
            layer = state.get("li", -1)
            if layer < 0 or layer >= 15:
                return tracer
            if frame.f_lineno == pretrigger:
                assert layer not in pre_seen, ("duplicate pretrigger", layer)
                pre_seen.add(layer)
                graph_capture.record_pre(state)
                def array(name):
                    return np.asarray(state[name], dtype=np.float64).copy()

                sigma, alpha, phi, cdf = map(array, ("sigma", "alpha", "phi", "Phi"))
                d3 = array("D3") if state["mode"] == 1 else np.zeros(1024)
                d21norm = (np.abs(np.asarray(state["D21"])).sum(axis=1, dtype=np.float64)
                           if state["mode"] == 1 else np.zeros(1024))
                g4 = array("g4row") if state["mode"] == 1 else np.zeros(1024)
                skew, kurt = d3 / sigma**3, g4 / sigma**4
                hermites = [np.ones(1024), alpha]
                for degree in range(1, 7):
                    hermites.append(alpha * hermites[-1] - degree * hermites[-2])
                values = [np.ones(1024), array("mu"), array("var"), sigma, alpha,
                          abs(alpha), phi, cdf, array("pk1v"), d3, d21norm,
                          np.full(1024, float(state["K4_sigma"])) if state["mode"] == 1 else np.zeros(1024),
                          sigma * phi, skew * hermites[2] * phi,
                          kurt * hermites[3] * phi, skew**2 * hermites[5] * phi,
                          skew * kurt * hermites[6] * phi, kurt**2 * hermites[7] * phi,
                          skew, kurt, cdf, np.full(1024, layer / 15)]
                features.append(np.stack(values, axis=1))
                scales.append(np.stack([sigma**p for p in range(1, 5)], axis=1))
            elif frame.f_lineno == posttrigger:
                assert layer not in post_seen, ("duplicate posttrigger", layer)
                post_seen.add(layer)
                delta = np.asarray(state["delta"]) if state["delta"] is not None else 0.0
                mean = np.asarray(state["K1v"], dtype=np.float64) + delta
                post = np.stack([mean] + [np.asarray(state[name], dtype=np.float64)
                                         for name in ("K2v", "K3v", "K4v")], axis=1)
                states.append(post.copy())
                inverse = np.asarray(state["inv_perm"]) if state["inv_perm"] is not None else np.arange(1024)
                if layer > 0:
                    # ORACLE offline reset; true labels permitted for diagnostic,
                    # never a candidate or learned estimator accuracy claim.
                    permutation = np.argsort(inverse)
                    updated = truth_post[layer][permutation].copy()
                    assert np.isfinite(updated).all()
                    fnp.copyto(state["K1v"], fnp.asarray((updated[:, 0]-delta).astype(np.float32)))
                    for channel, name in enumerate(("K2v", "K3v", "K4v"), 1):
                        fnp.copyto(state[name], fnp.asarray(updated[:, channel].astype(np.float32)))
                else:
                    updated = post.copy()
                after_states.append(updated[inverse])
                actual_mean = np.asarray(state["K1v"], dtype=np.float64)+delta
                actual_after_states.append(np.stack([actual_mean]+[np.asarray(state[name], dtype=np.float64)
                        for name in ["K2v", "K3v", "K4v"]], axis=1)[inverse])
                covariance_diagnostics.append(dict(layer=layer, min_diagonal=float(updated[:, 1].min()),
                      max_abs_marginal=float(np.max(abs(updated))), eigenvalues_not_computed=True))
                # Local labels are canonical original neuron IDs. Features used
                # by live roll-in above remain internal; saved training rows do not.
                graph_capture.canonicalize(layer,inverse)
                features[-1] = features[-1][inverse]
                scales[-1] = scales[-1][inverse]
                states[-1] = states[-1][inverse]
            return tracer

        with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
            mlp = SimpleNamespace(width=1024, weights=[fnp.asarray(w) for w in weights])
            sys.settrace(tracer)
            try:
                prediction = np.asarray(estimator.predict(mlp, 2**41)).copy()
            finally:
                sys.settrace(None)
        assert len(features) == len(states) == len(scales) == 15
        assert prediction.shape == (16, 1024) and np.isfinite(prediction).all()
        first_sigma = np.sqrt((weights[0].astype(np.float64)**2).sum(axis=0))
        analytic_first_mean = first_sigma / np.sqrt(2 * np.pi)
        analytic_first_variance = first_sigma**2 * (0.5 - 1 / (2 * np.pi))
        permute = module.PRUNE
        rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        address_limit = list(resource.getrlimit(resource.RLIMIT_AS))
        assert all(np.isfinite(value).all() for value in features + states + scales)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, idx=np.asarray(idx), features=np.stack(features),
                            states=np.stack(states), scales=np.stack(scales), prediction=prediction,
                            analytic_first_mean=analytic_first_mean,
                            analytic_first_variance=analytic_first_variance,
                            after_states=np.stack(after_states), actual_after_states=np.stack(actual_after_states),
                            graph_features=graph_capture.array(), local_features32=graph_capture.join_local32(np.stack(features),np.stack(states),np.stack(scales)))
        return {"complete": True, "idx": idx, "weight_seed": 770000 + idx,
                "weights_sha256": weights_sha, "h18_sha256": SOURCE_SHA,
                "elapsed_s": time.monotonic() - started, "cost": context.summary_dict(),
                "clipped_mean": clipped_mean, "clipped_var": clipped_var,
                "covariance_psd_diagnostic": covariance_diagnostics,
                "graph_capture_audit":graph_capture.audit, "permutation_flag": permute, "maxrss_kib": rss_kib,
                "rlimit_as": address_limit, "captured_pre_layers": sorted(pre_seen),
                "captured_post_layers": sorted(post_seen),
                "versions": {k: importlib.metadata.version(k) for k in
                             ("numpy", "flopscope", "whestbench", "torch")},
                "payload": buffer.getvalue()}
    except Exception as error:
        return {"complete": False, "idx": idx, "error_type": type(error).__name__,
                "error": str(error), "elapsed_s": time.monotonic() - started}


def score(prediction, truth):
    import numpy as np

    return float(np.mean((prediction.astype(np.float64) - truth.astype(np.float64))**2))


@app.local_entrypoint()
def main():
    raise RuntimeError("R50 template only; root must authorize immutable launch preparation before cloud use")
    import numpy as np

    parent = Path("results/phase2/modal_joint_closure_pilot_r38_r02")
    labels = np.load(parent/"research_labels.npz", allow_pickle=False)
    raw = labels["moments"][:24, :, :15].astype(np.float64)
    m1, m2, m3, m4 = [raw[:, i] for i in range(4)]
    variance = m2-m1*m1
    truth = np.stack([m1, variance, m3-3*m1*m2+2*m1**3,
                      m4-4*m1*m3+6*m1*m1*m2-3*m1**4-3*variance**2], axis=-1)
    baseline = {row["idx"]: row for row in map(json.loads, (parent/"journal.jsonl").read_text().splitlines())
                if row.get("complete") and row.get("phase") in ["baseline", "smoke"]}
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    OUT.mkdir(exist_ok=False)
    manifest = dict(version="teacher.pregraph.capture.r50", pid=os.getpid(),
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    anchor_source_sha256=SOURCE_SHA, indices=INDICES,
                    labels_sha256=hashlib.sha256((parent/"research_labels.npz").read_bytes()).hexdigest(),
                    memory_mib=16384, cpu_per_worker=1, max_workers=24,
                    first_layer="untouched known analytic", reset_layers=list(range(1, 15)),
                    scope="Offline TRAIN teacherforcedpregraphcapture with MC resets only to reconstructsameR48localtarget; no candidate/fitting/score/stage acceptance",
                    changes_vs_r48_capture=["alltrain24teacherforcedcapture; addedfixed6targetfreepregraphfeatures", "no denseeigen solve; no fitting/model/scoring", "features32=pre22+ownpost4/sigmapowers+graph6 beforecurrentreset"])
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    jobs = [(idx, baseline[idx]["weights_sha256"], truth[position].tobytes()) for position, idx in enumerate(INDICES)]
    with (OUT/"journal.jsonl").open("x") as journal:
        for row in rollout.map(jobs, order_outputs=False, return_exceptions=True):
            if isinstance(row, Exception):
                row = dict(complete=False, error_type=type(row).__name__, error=str(row))
            if row.get("complete"):
                payload = row.pop("payload")
                path = OUT/f"oracle_idx{row['idx']:04d}.npz"
                path.write_bytes(payload)
                row.update(path=str(path), payload_sha256=hashlib.sha256(payload).hexdigest())
                captured = np.load(io.BytesIO(payload), allow_pickle=False)
                assert captured['graph_features'].shape == (15,1024,6)
                assert captured['local_features32'].shape == (15,1024,32)
                row.update(feature_only_capture=True, no_final_reference_read=True)
            rows.append(row)
            journal.write(json.dumps(row)+"\n")
            journal.flush()
            print("PREGRAPH", row.get("idx"), row.get("complete"), flush=True)
    (OUT/"completion.json").write_text(json.dumps(dict(complete=True, rows=rows,
            failures=sum(not row["complete"] for row in rows), scope=manifest["scope"]), indent=2)+"\n")


if __name__ == "__main__":
    raise SystemExit("Launch with modal run; no local torch installation/download.")
