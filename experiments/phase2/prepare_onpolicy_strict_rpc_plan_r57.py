"""Write strict same-cycle paired RPC commands without executing them.

Only after real R56 model/source freeze and actual fresh raw evidence. Plan
uses official public benchmark protocol and current strict RPC runner limits.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--dataset", default="benchmark/data/mini_first8")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--n-mlps", type=int, default=1)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and sha(args.candidate) == args.candidate_sha256
    assert args.n_mlps > 0 and args.run_name.replace("_", "").isalnum()
    entries = json.loads(args.inputs.read_text())[:args.n_mlps]
    assert len(entries) == args.n_mlps
    for entry in entries:
        assert sha(Path(entry["weights"])) == entry["sha256"]
    baseline = Path("estimators/sprint_public_v37b_h18_r01/estimator.py")
    assert sha(baseline) == "c3dd3282106ba72f5882b2c2e1a0143c3627c373acaae0642c1e0033788ffda2"
    commands = []
    paths = {}
    for name, estimator in (("H18", baseline), ("R56", args.candidate)):
        prefix = f"results/phase2/{args.run_name}_{name}"
        local, rpc, scored = prefix+"_local.json", prefix+"_rpc.json", prefix+"_rpc_score.json"
        assert all(not Path(p).exists() for p in (local, rpc, scored))
        commands.extend([
            ["uv", "run", "--frozen", "python", "benchmark/phase2.py", "run", "--runner", "local",
             "--estimator", str(estimator), "--dataset", args.dataset,
             "--n-mlps", str(args.n_mlps), "--output", local],
            ["uv", "run", "--frozen", "python", "benchmark/run_rpc_strict.py", "--estimator", str(estimator),
             "--dataset", args.dataset, "--inputs", str(args.inputs),
             "--n-mlps", str(args.n_mlps), "--output", rpc],
            ["uv", "run", "--frozen", "python", "benchmark/rpc_strict_score.py", "--predictions", rpc,
             "--dataset", args.dataset, "--local", local, "--output", scored],
        ])
        paths[name] = dict(local=local, rpc=rpc, score=scored)
    commands.append(["uv", "run", "--frozen", "python", "benchmark/compare_strict_cohort.py",
                     "--before", paths["H18"]["score"], "--after", paths["R56"]["score"],
                     "--output", f"results/phase2/{args.run_name}_samecycle_strict_comparison.json"])
    report = dict(version="r57-strict-RPC-plan-only", candidate_sha256=args.candidate_sha256,
        parent_sha256=sha(baseline), commands=commands, executed=False,
        inputs_sha256=sha(args.inputs),
        runner_hashes={p: sha(Path(p)) for p in ("benchmark/phase2.py", "benchmark/run_rpc_strict.py",
            "benchmark/rpc_strict_worker.py", "benchmark/rpc_strict_score.py", "benchmark/compare_strict_cohort.py")},
        required_constraints=dict(flop_budget=2**41, lambda_flops_per_second=0, seed=0,
            thread_count=1, predict_wall_s=120, setup_timeout_s=5, residual_wall_s=.4,
            rpc_client_rlimit_as_bytes=8*1024**3),
        local_server_arraycap_bytes=1073741824,
        server_arraycap_official_deployment_confirmed=False,
        setup_RPC_guard_note="existing strict supervisor Session enforces5ssetup response/measuredelapsed; strictworker8GiBRLIMIT_AS; postpredict metered budget/time/residual gate",
        scope="strict commands plan, no acceptance/score/online claim; local16GiBcloud not substitute")
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
