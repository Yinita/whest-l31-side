"""One fixed TRAIN idx0 target-free observer versus actual candidate whole parity.

Instrumentation is offline and separately metered. No labels or scores read.
This is not resource/RPC/stage acceptance and never changes frozen R54.
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

BASELINE = Path("estimators/sprint_public_v37b_h18_r01/estimator.py")
BASELINE_SHA = "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
HELPERS = {
    "onpolicy_whole_observer_r59.py": "9745b146611b9d0fb9c5f7c3c2efb8ccd64bb6460745f9f8d1321d5f02ac715d",
    "tenfold_local_features_r48.py": "584626237242e84acb696aaf51b71724ad05db3007d70b475514819bc0b61bcf",
}
OUT = Path("results/phase2/onpolicy_whole_parity_r59")
app = modal.App("whest-onpolicy-whole-parity-r59")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "flopscope==0.12.1", "whestbench==0.16.1")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_file(str(BASELINE), "/code/h18.py")
)
for name in HELPERS:
    image = image.add_local_file(str(Path("experiments/phase2") / name), "/code/" + name)


@app.function(image=image, cpu=(1, 1), memory=(16384, 16384), timeout=600,
              retries=0, max_containers=1, scaledown_window=2)
def compare(job):
    import gc
    import importlib.metadata
    import importlib.util
    import platform
    import re
    import resource
    import sys
    import tempfile
    from types import SimpleNamespace

    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np
    import torch
    from whestbench import SetupContext

    assert set(job) == {"idx", "candidate_source", "candidate_sha256", "support_bytes",
                        "support_sha256", "support_metadata", "model_bytes", "model_sha256"}
    assert job["idx"] == 0
    started = time.monotonic()
    rows = []
    try:
        for key in tuple(os.environ):
            if re.match(r"V\d+_", key):
                os.environ.pop(key)
        assert hashlib.sha256(Path("/code/h18.py").read_bytes()).hexdigest() == BASELINE_SHA
        for name, digest in HELPERS.items():
            assert hashlib.sha256(Path("/code", name).read_bytes()).hexdigest() == digest
        for name in ("candidate", "support", "model"):
            content = job["candidate_source"] if name == "candidate" else job[name + "_bytes"]
            assert hashlib.sha256(content).hexdigest() == job[name + "_sha256"]
        model = json.loads(job["model_bytes"])
        assert model["train_indices"] == list(range(0, 384, 16))
        assert model["fitted_layers"] == list(range(1, 15))
        assert model["training_policy"] == "sequential_on_policy_frozen_previous_heads"
        metadata = json.loads(job["support_metadata"])
        assert metadata["model_sha256"] == job["model_sha256"]
        assert metadata["support_sha256"] == job["support_sha256"]
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu").manual_seed(770000)
        weights = (torch.randn(16, 1024, 1024, generator=generator, dtype=torch.float32)
                   * (2.0 / 1024)**.5).numpy()
        predictions = {}
        observer_audit = None
        with tempfile.TemporaryDirectory(prefix="r59-actual-support-") as temporary:
            folder = Path(temporary)
            candidate = folder / "estimator.py"
            candidate.write_bytes(job["candidate_source"])
            (folder / "coefficients.npz").write_bytes(job["support_bytes"])
            (folder / "coefficients.json").write_text(job["support_metadata"])
            sys.path.insert(0, "/code")
            from onpolicy_whole_observer_r59 import WholeObserver

            for phase, source in (("observer_instrumented", Path("/code/h18.py")),
                                  ("actual_candidate", candidate)):
                gc.collect()
                spec = importlib.util.spec_from_file_location("r59_" + phase, source)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                assert module.NO_CORR is True
                estimator = module.Estimator()
                with flops.BudgetContext(flop_budget=2**41, quiet=True) as setup_context:
                    estimator.setup(SetupContext(width=1024, depth=16, flop_budget=2**41,
                                                 api_version="0.1", seed=0))
                observer = None
                if phase == "observer_instrumented":
                    lines = source.read_text().splitlines()
                    preline = next(i for i, line in enumerate(lines, 1)
                                   if "if riders and not NO_CORR:" in line)
                    postline = next(i for i, line in enumerate(lines, 1)
                                    if "mu = K1v if delta is None else K1v + delta" in line)
                    observer = WholeObserver({int(layer): {k: fnp.asarray(v, dtype=fnp.float64)
                        for k, v in head.items()} for layer, head in model["layer_models"].items()})

                    def tracer(frame, event, arg):
                        if event == "line" and frame.f_code.co_filename == str(source):
                            if frame.f_lineno == preline:
                                observer.record_pre(frame.f_locals)
                            elif frame.f_lineno == postline:
                                observer.record_post(frame.f_locals)
                        return tracer

                with flops.BudgetContext(flop_budget=2**41, quiet=True) as context:
                    mlp = SimpleNamespace(width=1024, weights=[fnp.asarray(w) for w in weights])
                    if observer is not None:
                        sys.settrace(tracer)
                    try:
                        prediction = np.asarray(estimator.predict(mlp, 2**41)).copy()
                    finally:
                        sys.settrace(None)
                assert prediction.shape == (16, 1024) and np.isfinite(prediction).all()
                predictions[phase] = prediction
                if observer is not None:
                    observer_audit = observer.validate()
                rows.append(dict(phase=phase, cost=context.summary_dict(),
                    setup_cost=setup_context.summary_dict(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    no_corr_default=module.NO_CORR, output_dtype=str(prediction.dtype),
                    cumulative_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                del mlp, estimator, module
        difference = predictions["actual_candidate"].astype(np.float64) - predictions["observer_instrumented"].astype(np.float64)
        maximum = np.max(abs(difference), axis=1)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, idx=0, **predictions)
        return dict(complete=True, idx=0, rows=rows, observer_audit=observer_audit,
            maxabs_by_layer=maximum.tolist(), rms_by_layer=np.sqrt(np.mean(difference**2, axis=1)).tolist(),
            final_difference_mse=float(np.mean(difference[-1]**2)),
            passed=bool(maximum.max() < 1e-6), frozen_tolerance=1e-6,
            weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
            model_sha256=job["model_sha256"], candidate_sha256=job["candidate_sha256"],
            support_sha256=job["support_sha256"], same_worker_same_weights=True,
            labels_reference_absent=True, fixed_train_idx_zero=True,
            versions={k: importlib.metadata.version(k) for k in ("numpy", "flopscope", "whestbench", "torch")},
            processor=platform.processor(), elapsed_s=time.monotonic()-started,
            payload=buffer.getvalue())
    except Exception as error:
        return dict(complete=False, idx=0, rows=rows, error_type=type(error).__name__,
                    error=str(error), elapsed_s=time.monotonic()-started)


@app.local_entrypoint()
def main(candidate_dir: str, model_path: str, model_sha256: str,
         candidate_sha256: str, support_sha256: str):
    assert not OUT.exists()
    directory, model_file = Path(candidate_dir), Path(model_path)
    for name, digest in HELPERS.items():
        assert hashlib.sha256(Path("experiments/phase2", name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA
    job = dict(idx=0, candidate_source=(directory / "estimator.py").read_bytes(),
        candidate_sha256=candidate_sha256, support_bytes=(directory / "coefficients.npz").read_bytes(),
        support_sha256=support_sha256, support_metadata=(directory / "coefficients.json").read_text(),
        model_bytes=model_file.read_bytes(), model_sha256=model_sha256)
    for name in ("candidate", "support", "model"):
        content = job["candidate_source"] if name == "candidate" else job[name + "_bytes"]
        assert hashlib.sha256(content).hexdigest() == job[name + "_sha256"]
    OUT.mkdir()
    manifest = dict(version="r59-fixedidx0-targetfree-full16-observer-actual-parity", pid=os.getpid(),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_hashes=HELPERS, baseline_sha256=BASELINE_SHA,
        model_sha256=model_sha256, candidate_sha256=candidate_sha256, support_sha256=support_sha256,
        idx=0, seed=0, weight_seed=770000, threads=1, retries=0,
        memory_mib=16384, frozen_tolerance=1e-6,
        phase_order=["observer_instrumented", "actual_candidate"],
        scope="target-free whole16implementation parity; instrumentation separatelymetered; no score/RPC/resource/stage acceptance")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = compare.remote(job)
    if result.get("complete"):
        payload = result.pop("payload")
        path = OUT / "private_predictions_idx0000.npz"
        path.write_bytes(payload)
        result.update(payload_sha256=hashlib.sha256(payload).hexdigest(), path=str(path))
    result["scope"] = manifest["scope"]
    (OUT / "completion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
