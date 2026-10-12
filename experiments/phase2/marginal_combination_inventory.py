"""Exact scalar cumulant expansion inventory, separate from strict K3 closure."""
import json
import math
from fractions import Fraction
from pathlib import Path


def main():
    existing = json.loads(Path('results/phase2/ablation_v30_033_inventory.json').read_text())
    rows = []
    for degree in range(3, 9):
        for a in range(3):
            for b in range(3):
                for c in range(2):
                    for d in range(2):
                        powers = (a, b, c, d)
                        if sum(k * n for k, n in zip((3, 4, 5, 6), powers)) != degree:
                            continue
                        coef = Fraction(1, math.prod(math.factorial(n) * math.factorial(k)**n for k, n in zip((3, 4, 5, 6), powers)))
                        factors = ['d3'] * a + ['g4'] * b + ['k5'] * c + ['k6'] * d
                        for power in range(1, 5):
                            terms = existing['rows'][power-1]['terms']
                            present = any(sorted(x['factors']) == sorted(factors + ['ones1'] * (2-len(factors))) and x['wick'][0] == degree for x in terms) if len(factors) <= 2 else False
                            rows.append(dict(output_power=power, derivative_degree=degree, factors=factors, coefficient=str(coef), present_v29=present,
                                             note='Higher derivative expansion; not strict K3 diagram budget'))
    result = dict(scope='Scalar exp(sum kappa_r derivative^r/r!), derivative degree <=8; not exhaustive multiaxis diagrams', rows=rows,
                  known_negative='k4^2 mean candidate previously failed adjusted gate; not rerun',
                  policy='k5/k6 single terms and same derivative-order k3^2 must be reported separately; no score from inventory')
    path = Path('results/phase2/ablation_v30_086_marginal_inventory.json')
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2)+'\n')
    print('rows', len(rows))


if __name__ == '__main__':
    main()
