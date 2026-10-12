#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_102_r03_singlethread_g4_cv.json; do sleep 5; done
uv run --frozen python - <<'PY'
import json
from pathlib import Path
report='results/phase2/ablation_v30_102_r03_singlethread_g4_cv.json';r=json.loads(Path(report).read_text());p=Path('experiments/phase2/ablation_queue.json');q=json.loads(p.read_text());t=next(t for t in q['tasks'] if t['key']=='L07');t['evidence']+=['results/phase2/ablation_v30_102_r03_singlethread_g4_features.json',report];t['singlethread_samecycle_cv']=r;t['result']['conclusion']='Singlethread actual g4 trace and samecycle13/15feature networkLOO completed; no charged finalcandidate; negative L03 reliableparent gate retained';p.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
with Path('experiments/phase2/RESEARCH_LOG.md').open('a') as f:f.write('\nL07 singlethread samecycleCV '+json.dumps(r)+'\n')
PY
uv run --frozen python experiments/phase2/update_ablation_results.py
