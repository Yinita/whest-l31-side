"""Every queue item links its evidence; missing evidence remains explicit."""
import hashlib
import json
from pathlib import Path


def main():
    q = json.loads(Path('experiments/phase2/ablation_queue.json').read_text())
    rows = []
    for t in q['tasks']:
        evidence = []
        for name in t.get('evidence', []):
            p = Path(name)
            evidence.append(dict(path=name, exists=p.exists(), sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None))
        rows.append(dict(key=t['key'], version=t['version'], status=t['status'], evidence=evidence))
    output = Path('results/phase2/ablation_v30_129_r05_queue_evidence_index.json')
    assert not output.exists()
    output.write_text(json.dumps(dict(rows=rows, scope='129 task to evidence mapping; report counts are not effective independent trials; selection exposure cannot be reconstructed beyond saved logs', confirmation='Public8-15 labels and state opened, classify as development/publicconfirmation; newseed stronglabels absent; no final independent set accepted'),indent=2)+'\n')


if __name__ == '__main__':
    main()
