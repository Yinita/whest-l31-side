"""Bounded static extraction of the official pure Python diagram generators.

No Torch algorithm runs here. Python 3.12 type statements are stripped solely
for Python 3.10 parsing, then only named combinatorial definitions are executed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import logging
import math
import platform
import subprocess
from collections import Counter, OrderedDict, defaultdict
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np

OFFICIAL_SHA = "93d091a4c26c042bfffa28f2e76a81bc0aba94bb"
GENERATORS = {
    "get_int_cond",
    "vec_block_cost",
    "get_vec_cond",
    "is_hypertree",
    "factored_keeps_term",
    "get_all_terms",
    "get_all_terms_iso",
}


class QuietProgress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


def extract_generators(repo: Path):
    paths = [repo / "src/mlp_kprop/partitions.py", repo / "src/mlp_kprop/kprop_harmonic.py"]
    ns = {
        "__name__": "official_combinatorics_extracted",
        "math": math,
        "itertools": itertools,
        "np": np,
        "defaultdict": defaultdict,
        "Counter": Counter,
        "OrderedDict": OrderedDict,
        "cache": cache,
        "Any": Any,
        "Callable": Callable,
        "Iterable": Iterable,
        "Iterator": Iterator,
        "Optional": Optional,
        "Self": Any,
        "combinations": itertools.combinations,
        "product": itertools.product,
        "logger": logging.getLogger("official_combinatorics_audit"),
        "logging": logging,
        "tqdm": QuietProgress,
    }
    for i, path in enumerate(paths):
        # This adapts annotation syntax; function bodies and decorators are unchanged.
        source = "\n".join(
            line for line in path.read_text().splitlines() if not line.startswith("type ")
        )
        tree = ast.parse(source)
        defs = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and (i == 0 or node.name in GENERATORS)
        ]
        module = ast.Module(body=defs, type_ignores=[])
        future = ast.parse("from __future__ import annotations").body[0]
        module.body.insert(0, future)
        ast.fix_missing_locations(module)
        exec(compile(module, str(path), "exec"), ns)
    return ns, [
        {"path": str(p.relative_to(repo)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in paths
    ]


def coefficient(part, count):
    denom = math.prod(
        math.factorial(c) * math.prod(math.factorial(x) for x in v) ** c
        for v, c in Counter(part).items()
    )
    return Fraction(count, denom)


def factor_info(vec):
    nonzero = tuple(sorted((x for x in vec if x), reverse=True))
    if sum(vec) == 2:
        name = "c_off"
    elif sum(vec) == 3:
        name = "d3" if nonzero == (3,) else "d21" if nonzero == (2, 1) else "factored_k111"
    elif sum(vec) == 4:
        name = (
            "g4"
            if nonzero == (4,)
            else "wk4m"
            if nonzero == (2, 2)
            else "wk431"
            if nonzero == (3, 1)
            else "harmonic_k211"
        )
    else:
        name = "unavailable"
    return {
        "vsum": sum(vec),
        "vector": vec,
        "slice": nonzero,
        "v29_slice": name,
        "available_dense_v29": name in {"c_off", "d3", "d21", "g4", "wk4m", "wk431"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-repo", type=Path, default=Path("/tmp/whest-official-cumulant-reference")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve existing audit evidence"
    sha = subprocess.check_output(
        ["git", "-C", str(args.official_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert sha == OFFICIAL_SHA, (sha, OFFICIAL_SHA)
    ns, sources = extract_generators(args.official_repo)
    before = ns["get_all_terms_iso"](3, d_max=4, augment=False)
    after = ns["get_all_terms_iso"](3, d_max=4, augment=True)
    v29_tree = ast.parse(Path("estimators/k3_v29/estimator.py").read_text())
    wick_pairs = next(
        ast.literal_eval(node.value)
        for node in v29_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "WICK_PAIRS" for t in node.targets)
    )
    term_specs = next(
        ast.literal_eval(node.value)
        for node in v29_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TERM_SPECS" for t in node.targets)
    )

    def signature(names, wicks):
        return tuple(sorted(names)), tuple(tuple(w) for w in wicks)

    v29_terms = {}
    for ip, entries in term_specs.items():
        v29_terms[ip] = {
            signature((a, b), (wick_pairs[wl], wick_pairs[wr])): coef
            for a, b, wl, wr, coef in entries
            if wr is not None
        }
    for ip, wl, wr, coef in [
        ((1, 1), (1, 1), (3, 1), 1 / 3),
        ((2, 1), (1, 2), (3, 1), 1 / 6),
        ((2, 1), (3, 2), (1, 1), 1 / 6),
        ((2, 2), (1, 2), (3, 2), 1 / 3),
    ]:
        v29_terms[ip][signature(("wk431", "ones2"), (wl, wr))] = coef

    def actual_name(v):
        if v == (1, 1):
            return "c_off"
        if v == (0, 3):
            return "d3row"
        if v == (3, 0):
            return "d3col"
        if v == (0, 4):
            return "g4row"
        if v == (4, 0):
            return "g4col"
        if v == (2, 1):
            return "d21"
        if v == (1, 2):
            return "d21T"
        if v == (2, 2):
            return "wk4m"
        if v in ((3, 1), (1, 3)):
            return "wk431"
        return "unmapped"

    extras = []
    all_counts = {}
    for ip in sorted(after, key=lambda x: (sum(x), len(x), x)):
        all_counts[str(ip)] = {"base": len(before.get(ip, {})), "augment": len(after[ip])}
        for vp, count in sorted(after[ip].items()):
            if vp in before.get(ip, {}):
                assert count == before[ip][vp]
                continue
            c = coefficient(vp, count)
            assert math.isclose(
                float(c), count * ns["vec_part_coef"](vp, divide_fac=True), rel_tol=1e-15
            )
            axes = ns["check_vec_partition"](vp, len(ip))
            wicks = list(zip(axes, ip))
            factors = [factor_info(v) for v in vp]
            category = (
                "dense"
                if len(ip) <= 2
                else "capped211"
                if ip == (2, 1, 1)
                else "factored111"
                if ip == (1, 1, 1)
                else "other"
            )
            names = [actual_name(v) for v in vp] if len(ip) == 2 else []
            if len(names) == 1:
                names.append("ones2")
            existing_coef = v29_terms.get(ip, {}).get(signature(names, wicks))
            extras.append(
                {
                    "v29_operand_names": names,
                    "v29_output_present": ip in ((1, 1), (2, 1), (2, 2)),
                    "already_present_in_v29": existing_coef is not None,
                    "existing_v29_coefficient": existing_coef,
                    "matching_coefficient": existing_coef is not None
                    and math.isclose(existing_coef, float(c), rel_tol=1e-15),
                    "output_part": ip,
                    "vector_partition": vp,
                    "iso_count": count,
                    "exact_coefficient": str(c),
                    "coefficient": float(c),
                    "wick_kp": wicks,
                    "factors": factors,
                    "category": category,
                    "official_factored_keeps": ns["factored_keeps_term"](3, ip, vp),
                    "available_v29_slices": all(f["available_dense_v29"] for f in factors),
                    "available_v29_wicks": all(tuple(w) in wick_pairs for w in wicks),
                    "power_count_cost": sum(ns["vec_block_cost"](v) for v in vp),
                }
            )
    result = {
        "official_repository": "https://github.com/alignment-research-center/mlp_cumulant_propagation",
        "official_commit": sha,
        "license": "MIT Copyright 2026 Alignment Research Center",
        "mode": "AST-extracted unchanged combinatorial definitions; annotation syntax adapted for Python 3.10; no Torch/nonlinear algorithm execution",
        "sources": sources,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fixed_generator_arguments": {"k_max": 3, "d_max": 4, "use_mean_var": False},
        "counts_by_output": all_counts,
        "extra_counts": dict(Counter(x["category"] for x in extras)),
        "ready_dense_count": sum(
            x["category"] == "dense" and x["available_v29_slices"] and x["available_v29_wicks"]
            for x in extras
        ),
        "missing_ready_existing_output_terms": [
            x
            for x in extras
            if x["category"] == "dense"
            and x["v29_output_present"]
            and not x["already_present_in_v29"]
            and x["available_v29_slices"]
            and x["available_v29_wicks"]
        ],
        "terms": extras,
        "limitations": [
            "Does not demonstrate numerical production parity or score improvement.",
            "Adding a subset is a mechanism ablation, not the full augmented algorithm.",
            "Full factored AUGMENT drops non-hypertree and top-cumulant cross-edge terms.",
            "The public lean aug generator does not pass augment=True to get_all_terms_iso.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("official_commit", "extra_counts", "ready_dense_count")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
