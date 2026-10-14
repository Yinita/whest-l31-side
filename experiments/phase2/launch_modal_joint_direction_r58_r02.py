"""Launch authorized frozen R58 once; keep credentials private."""
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

OUT=Path('results/phase2/joint_direction_mc_r58_r02')
PREPARATION=Path('results/phase2/joint_direction_preparation_r58_r04.json')

root=Path(__file__).resolve().parents[2]
preparation=json.loads((root/PREPARATION).read_text())
for path,expected in preparation['source_sha256'].items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected
env=os.environ.copy()
for line in Path('/home/sti/Desktop/keys.env').read_text().splitlines():
    line=line.strip()
    if line.startswith('export '):line=line[7:]
    key,separator,value=line.partition('=')
    if separator and key.strip() in {'MODAL_TOKEN_ID','MODAL_TOKEN_SECRET'}:
        parts=shlex.split(value,comments=True)
        if parts:env[key.strip()]=parts[0]
assert 'MODAL_TOKEN_ID' in env and 'MODAL_TOKEN_SECRET' in env
log=root/'results/phase2/joint_direction_mc_r58_r02.launcher.log'
assert not log.exists() and not (root/OUT).exists()
with log.open('xb') as stream:
    process=subprocess.Popen(['/home/sti/.local/bin/modal','run','experiments/phase2/modal_joint_direction_mc_r58_r02.py'],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=process.pid,log=str(log),source_sha256=preparation['source_sha256']['experiments/phase2/modal_joint_direction_mc_r58_r02.py'],smoke_first=True)))
