"""Privately load existing authorized Modal credentials; frozen R59 only."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
env = os.environ.copy()
for line in Path("/home/sti/Desktop/keys.env").read_text().splitlines():
    line = line.strip()
    if line.startswith("export "):
        line = line[7:]
    key, separator, value = line.partition("=")
    if separator and key.strip() in {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}:
        fields = shlex.split(value, comments=True)
        if fields:
            env[key.strip()] = fields[0]
assert "MODAL_TOKEN_ID" in env and "MODAL_TOKEN_SECRET" in env
parser = argparse.ArgumentParser()
for name in ("candidate-dir", "model-path", "model-sha256", "candidate-sha256", "support-sha256"):
    parser.add_argument("--" + name, required=True)
args = parser.parse_args()
log = root / "results/phase2/modal_onpolicy_whole_parity_r59.launcher.log"
assert not log.exists() and not (root / "results/phase2/onpolicy_whole_parity_r59").exists()
command = ["/home/sti/.local/bin/modal", "run", "experiments/phase2/modal_onpolicy_whole_parity_r59.py",
           "--candidate-dir", args.candidate_dir, "--model-path", args.model_path,
           "--model-sha256", args.model_sha256, "--candidate-sha256", args.candidate_sha256,
           "--support-sha256", args.support_sha256]
with log.open("xb") as stream:
    process = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
print(f"R59 fixedidx0 targetfreewhole parity PID={process.pid} log={log}")
