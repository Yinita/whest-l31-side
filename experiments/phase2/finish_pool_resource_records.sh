#!/usr/bin/env bash
set -euo pipefail
cd /home/sti/Desktop/projects/whest-l31-side
while test ! -s results/phase2/ablation_v30_026_r03_shape_resource.json; do sleep 5; done
uv run --frozen python - <<'PY'
import json
from pathlib import Path
report='results/phase2/ablation_v30_026_r03_shape_resource.json'
r=json.loads(Path(report).read_text())
p=Path('experiments/phase2/ablation_queue.json');q=json.loads(p.read_text());t=next(t for t in q['tasks'] if t['key']=='C26');t['evidence'].append(report);t['result']['conclusion']='Actual shape sequence predict under8GiB cap completed; per-shape failure/RSS/VSZ/logical alias bytes retained. RPC same-shape lifetime still outstanding, nonfinite shape diagnostic warnings not numerical acceptance';p.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
with Path('experiments/phase2/RESEARCH_LOG.md').open('a') as f:f.write('\nC26 actual shape resource rows '+json.dumps([{k:v for k,v in row.items() if k!='pool_arrays'} for row in r['rows']])+'\n')
PY
uv run --frozen python experiments/phase2/update_ablation_results.py
