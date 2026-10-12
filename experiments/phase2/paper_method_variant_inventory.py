"""Bounded full-generator variant inventory; no numerical method score."""
import json
import subprocess
from pathlib import Path

from experiments.phase2.audit_official_augmented_terms import OFFICIAL_SHA, extract_generators


def main():
    repo = Path('/tmp/whest-official-cumulant-reference')
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == OFFICIAL_SHA
    ns, hashes = extract_generators(repo)
    rows = []
    for order in (2, 3, 4):
        for augmented in (False, True):
            terms = ns['get_all_terms_iso'](order, d_max=4, augment=augmented)
            dense = sum(len(v) for v in terms.values())
            retained = (sum(ns['factored_keeps_term'](order, output, part)
                            for output, table in terms.items() for part in table)
                        if order >= 3 else dense) # factor path only exists for K3/K4
            rows.append(dict(order=order, augmented=augmented,
                             bounded_dense_terms=dense, bounded_factor_retained=retained,
                             counts_by_output={str(k): len(v) for k, v in terms.items()},
                             scope='output degree<=4 only; not complete higher-order state inventory'))
    out = Path('results/phase2/paper_m03_r01_variant_inventory.json')
    assert not out.exists()
    out.write_text(json.dumps(dict(official_commit=OFFICIAL_SHA, sources=hashes, rows=rows,
                                  conclusion='Frozen variant counts expose pruning; no FLOP/runtime or numerical equivalence claim',
                                  adjusted_score=None), indent=2) + '\n')
    print(rows)


if __name__ == '__main__':
    main()
