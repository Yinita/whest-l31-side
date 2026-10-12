"""L20 frozen newseed network and reference protocol, no weak acceptance."""

import hashlib
import json
from pathlib import Path

import whestbench.generation as generation


def main():
    source = Path(generation.__file__)
    manifest = dict(
        width=1024,
        depth=16,
        weight_distribution="N(0,2/width),float32, officialsample_mlp",
        weight_rng_seeds=[12026001 + i for i in range(8)],
        grader_seeds=[22026001 + i for i in range(8)],
        reference_rng_seeds=[[32026001 + 2 * i, 32026002 + 2 * i] for i in range(8)],
        reference_samples_required=10**9,
        threads=1,
        generation_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        candidate_selection="FrozenMini8 chosen candidate must precede reference-label scoring; weight/grader streams distinct",
        status="protocolonly; networks and references not yet generated",
        limitation="StrongCPUreferences budgetnotestablished; smallpilot cannot grant independent10ximprovement. No paidcloud authorized.",
    )
    path = Path("results/phase2/ablation_v30_115_independent_protocol.json")
    assert not path.exists()
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
