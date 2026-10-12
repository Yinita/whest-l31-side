"""Prepare immutable V29 combined thin-leg transport experiment."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parent = root / 'estimators/k3_v29'
dest = root / 'estimators/ablation_v30_003'
dest.mkdir(exist_ok=True)
assert not (dest / 'estimator.py').exists()
source = (parent / 'estimator.py').read_text()
old = '''                Z_st = fnp.matmul(WDb, Z_st, out=pool.get(("z", z_side), (L - 1, n, r + 2))[:k])
                if Zf_st is not None:
                    Zf_st = fnp.matmul(WDb, Zf_st, out=pool.get(("zf", z_side), (L - 1, n, 2 * rfb))[:k])'''
new = '''                if Zf_st is not None:
                    # C03: one pooled transport; assembly and copies remain charged.
                    combined = pool.get(("z_combined_in", z_side), (L - 1, n, r + 2 + 2 * rfb))[:k]
                    fnp.copyto(combined[:, :, :r + 2], Z_st)
                    fnp.copyto(combined[:, :, r + 2:], Zf_st)
                    transported = fnp.matmul(WDb, combined, out=pool.get(("z_combined_out", z_side), (L - 1, n, r + 2 + 2 * rfb))[:k])
                    Z_st = pool.get(("z", z_side), (L - 1, n, r + 2))[:k]
                    Zf_st = pool.get(("zf", z_side), (L - 1, n, 2 * rfb))[:k]
                    fnp.copyto(Z_st, transported[:, :, :r + 2])
                    fnp.copyto(Zf_st, transported[:, :, r + 2:])
                else:
                    Z_st = fnp.matmul(WDb, Z_st, out=pool.get(("z", z_side), (L - 1, n, r + 2))[:k])'''
assert source.count(old) == 1
candidate = source.replace(old, new)
(dest / 'estimator.py').write_text(candidate)
shutil.copy(parent / 'LICENSE', dest / 'LICENSE')
provenance = json.loads((parent / 'provenance.json').read_text())
provenance.update(parent_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                  ablation_key='C03', ablation_version='v30.003',
                  modifications=['Combine Z/Zf transport in persistent pooled slab; explicit charged copyto'])
(dest / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
