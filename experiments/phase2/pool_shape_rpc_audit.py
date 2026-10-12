"""Reproduce the local official-client RPC probe in separate environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import flopscope.numpy as fnp
import whestbench
from whestbench import load_dataset, mlp_at

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="benchmark/data/mini_first8")
    parser.add_argument("--estimator", default="estimators/k3_v29/estimator.py")
    parser.add_argument("--n-mlps", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.n_mlps <= 0:
        parser.error("--n-mlps must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Use a fresh output filename")
    client_env = ROOT / "benchmark/data/client-venv"
    if not (client_env / "bin/python").exists():
        subprocess.run(["uv", "venv", str(client_env), "--python", "3.10"], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(client_env / "bin/python"),
            "-r",
            str(ROOT / "benchmark/client-requirements.txt"),
        ],
        check=True,
    )
    data = load_dataset(args.dataset, split="mini")
    # A run owns its inputs: other probes cannot overwrite files while a client reads.
    inputs = ROOT / "benchmark/data/rpc_inputs" / output.stem
    inputs.mkdir(parents=True, exist_ok=False)
    entries = []
    full = mlp_at(data, 0)
    for i, (width, depth) in enumerate(((64, 4), (1024, 16), (64, 4), (1024, 16))):
        file = inputs / f"mlp_{i}.npz"
        fnp.savez(str(file), weights=fnp.stack([w[:width, :width] for w in full.weights[:depth]]))
        with zipfile.ZipFile(file) as archive:
            corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"CRC failure {i}: {corrupt}")
        entries.append({"index": i, "name": full.name, "seed": full.seed,
                        "width": width, "depth": depth, "weights": str(file),
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    (inputs / "inputs.json").write_text(json.dumps(entries, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="whest-rpc-") as tmp:
        url = f"ipc://{tmp}/server.sock"
        env = os.environ.copy()
        env.update(
            {
                "FLOPSCOPE_MAX_ARRAY_BYTES": "1073741824",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        for key in list(env):
            if key.startswith(tuple(f"V{i}_" for i in range(17, 30))):
                del env[key]
        with output.with_suffix(".server.log").open("w") as server_log:
            server = subprocess.Popen(
                [sys.executable, "-m", "flopscope_server", "--url", url, "--timeout", "180"],
                env=env,
                stdout=server_log,
                stderr=server_log,
            )
            try:
                deadline = time.monotonic() + 10
                while not Path(tmp, "server.sock").exists():
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("RPC server did not start")
                    time.sleep(0.1)
                env["FLOPSCOPE_SERVER_URL"] = url
                command = [
                    str(client_env / "bin/python"),
                    "-u",
                    str(ROOT / "experiments/phase2/pool_shape_rpc_worker.py"),
                    "--sdk-root",
                    str(Path(whestbench.__file__).parent),
                    "--estimator",
                    str(Path(args.estimator).resolve()),
                    "--inputs",
                    str(inputs / "inputs.json"),
                    "--output",
                    str(output),
                    "--n-mlps",
                    str(args.n_mlps),
                ]
                with output.with_suffix(".client.log").open("w") as client_log:
                    subprocess.run(
                        command,
                        env=env,
                        stdout=client_log,
                        stderr=client_log,
                        check=True,
                        timeout=args.n_mlps * 150 + 30,
                    )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    print("Saved RPC predictions:", output)


if __name__ == "__main__":
    main()
