"""Locate target mixed diagram relative to official filter and actual V29 feedback."""

import hashlib
import json
from pathlib import Path

from experiments.phase2.audit_official_augmented_terms import extract_generators


def main():
    path = Path("results/phase2/ablation_v30_059_r03_mixed_coverage_audit.json")
    assert not path.exists()
    ns, sources = extract_generators(Path("/tmp/whest-official-cumulant-reference"))
    output = (1, 1, 1)
    partition = ((0, 1, 1), (1, 1, 1))
    classes = ns["get_all_terms_iso"](3, d_max=4, augment=True)[output]
    assert partition in classes
    assert not ns["factored_keeps_term"](3, output, partition)
    source = Path("estimators/k3_v29/estimator.py")
    text = source.read_text()
    evidence = {}
    for marker in ("Xt = fnp.einsum", "Yt = fnp.einsum", "X1 = 3 a_b + Xt", "LPadd = A*Yt"):
        evidence[marker] = [
            dict(line=i, text=line.strip())
            for i, line in enumerate(text.splitlines(), 1)
            if marker in line
        ]
    result = dict(
        output=output,
        partition=partition,
        official_orbit_count=classes[partition],
        official_factored_keeps=False,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_evidence=evidence,
        official_sources=sources,
        conclusion="Target K111*covariance orbit is explicitly excluded by official factored filter. V29 feedback constructs D21/covariance products and transports thin legs; that alone does not prove mixed full K111 is present. Birth mapping of all sources and raw third-state factors must be audited before restoration.",
        scope="Static source coverage triage; missing in official filter proven, V29 whole-state overlap not yet resolved",
        score=None,
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("Official excludes", partition, "V29 overlap still unresolved")


if __name__ == "__main__":
    main()
