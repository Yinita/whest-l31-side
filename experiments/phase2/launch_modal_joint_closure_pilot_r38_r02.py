"""Load authorized Modal credentials privately and launch one bounded pilot."""

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
log = root / "results/phase2/modal_joint_closure_pilot_r38_r02.launcher.log"
assert not log.exists() and not (root / "results/phase2/modal_joint_closure_pilot_r38_r02").exists()
with log.open("xb") as stream:
    process = subprocess.Popen(["/home/sti/.local/bin/modal", "run", "experiments/phase2/modal_joint_closure_pilot_r38_r02.py"],
                               cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True)
print(f"Joint closure corpus r38r02 PID={process.pid} log={log}")
