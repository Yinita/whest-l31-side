"""Regression evidence for strict RPC scoring, using frozen real V29 rows."""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from whestbench import load_dataset

from benchmark import rpc_strict_score

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmark/data/mini_first8"


@pytest.fixture(scope="module")
def existing_evidence():
    predictions = ROOT / "results/phase2/v29_mini8_rpc_predictions.json"
    local = ROOT / "results/phase2/v29_mini8_local.json"
    old = ROOT / "results/phase2/v29_mini8_rpc.json"
    if not predictions.exists() or not local.exists() or not DATASET.exists():
        pytest.skip("Frozen local benchmark evidence is unavailable")
    rpc = json.loads(predictions.read_text())
    rpc["manifest"].update(
        n_mlps=8, dataset=str(DATASET), setup_ok=True,
        dataset_commit=json.loads((DATASET / "provenance.json").read_text())["revision"],
        limits={"flop_budget": 2**41, "predict_seconds": 120,
                "residual_seconds": 0.4, "setup_seconds": 5},
    )
    for row in rpc["per_mlp"]:
        row.update(status="ok", failure=None)
    targets = load_dataset(str(DATASET), split="mini").select_columns(
        ["mlp_name", "mlp_seed", "all_layer_means", "final_means"]
    )
    return rpc, local, json.loads(old.read_text()), targets


def score(monkeypatch, tmp_path, evidence, *, local=None):
    source = tmp_path / "strict.json"
    output = tmp_path / "scored.json"
    source.write_text(json.dumps(evidence))
    argv = ["rpc_strict_score", "--predictions", str(source), "--dataset", str(DATASET),
            "--output", str(output)]
    if local is not None:
        argv += ["--local", str(local)]
    monkeypatch.setattr(sys, "argv", argv)
    rpc_strict_score.main()
    return json.loads(output.read_text())


def copy_local(tmp_path, path):
    copied = tmp_path / "local.json"
    copied.write_bytes(path.read_bytes())
    copied.with_suffix(".manifest.json").write_bytes(path.with_suffix(".manifest.json").read_bytes())
    return copied


def test_real_v29_success_matches_old_score(monkeypatch, tmp_path, existing_evidence):
    rpc, local, old, _ = existing_evidence
    result = score(monkeypatch, tmp_path, deepcopy(rpc), local=local)
    assert result["n_failed_mlps"] == 0
    assert result["final_layer_mse"] == old["final_layer_mse"]
    assert result["adjusted_final_layer_score"] == old["adjusted_final_layer_score"]
    assert result["local_comparison"]["same_candidate_parity_verified"] is True
    assert all(row["local_comparison_status"] == "both_passed" for row in result["per_mlp"])


@pytest.mark.parametrize("failure", ["status", "shape", "nan", "overflow", "flops", "wall", "residual", "invalid_summary", "missing_summary", "setup"])
def test_each_failure_is_retained_without_discount(monkeypatch, tmp_path, existing_evidence, failure):
    rpc, _, _, targets = existing_evidence
    evidence = deepcopy(rpc)
    row = evidence["per_mlp"][0]
    if failure == "status":
        row.update(status="worker_killed", failure={"reason": "timeout"}, predictions=None)
    elif failure == "shape":
        row["predictions"] = [[1.0]]
    elif failure == "nan":
        row["predictions"][0][0] = float("nan")
    elif failure == "overflow":
        row["predictions"][-1][0] = 1e38
    elif failure == "invalid_summary":
        row["budget_summary"]["wall_time_s"] = float("nan")
    elif failure == "missing_summary":
        row["budget_summary"] = None
    elif failure == "setup":
        evidence["manifest"]["setup_ok"] = False
    else:
        key, value = {"flops": ("flops_used", 2**41 + 1),
                      "wall": ("wall_time_s", 120.01),
                      "residual": ("residual_wall_time_s", 0.401)}[failure]
        row["budget_summary"][key] = value
    result = score(monkeypatch, tmp_path, evidence)
    expected = float(np.mean(np.asarray(targets[0]["final_means"], dtype=np.float32) ** 2))
    first = result["per_mlp"][0]
    assert len(result["per_mlp"]) == 8
    assert result["n_failed_mlps"] == (8 if failure == "setup" else 1)
    assert first["zero_fallback"] is True
    assert first["score_multiplier"] == 1.0
    assert first["final_layer_mse"] == expected
    assert first["adjusted_final_layer_score"] == expected
    if failure == "status":
        assert first["failure"] == {"reason": "timeout"}


@pytest.mark.parametrize("mistake", ["missing", "duplicate", "seed", "name"])
def test_invalid_network_protocol_rejected(monkeypatch, tmp_path, existing_evidence, mistake):
    evidence = deepcopy(existing_evidence[0])
    if mistake == "missing":
        evidence["per_mlp"].pop()
    elif mistake == "duplicate":
        evidence["per_mlp"][1]["mlp_index"] = 0
    elif mistake == "seed":
        evidence["per_mlp"][0]["seed"] += 1
    else:
        evidence["per_mlp"][0]["mlp_name"] = "different-network"
    with pytest.raises(ValueError):
        score(monkeypatch, tmp_path, evidence)


@pytest.mark.parametrize("mistake", ["candidate", "report"])
def test_wrong_local_hash_cannot_claim_parity(monkeypatch, tmp_path, existing_evidence, mistake):
    rpc, original_local, _, _ = existing_evidence
    local = copy_local(tmp_path, original_local)
    manifest_path = local.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["estimator_sha256" if mistake == "candidate" else "report_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    result = score(monkeypatch, tmp_path, deepcopy(rpc), local=local)
    assert result["n_failed_mlps"] == 0
    assert result["local_comparison"]["same_candidate_parity_verified"] is False
    assert result["local_comparison"]["verification_errors"]
    assert all("local_comparison_status" not in row for row in result["per_mlp"])


def test_local_failure_is_explicit_when_rpc_passes(monkeypatch, tmp_path, existing_evidence):
    rpc, original_local, _, _ = existing_evidence
    local = copy_local(tmp_path, original_local)
    document = json.loads(local.read_text())
    document["results"]["per_mlp"][0]["time_exhausted"] = True
    local.write_text(json.dumps(document))
    manifest_path = local.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["report_sha256"] = hashlib.sha256(local.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result = score(monkeypatch, tmp_path, deepcopy(rpc), local=local)
    assert result["local_comparison"]["same_candidate_parity_verified"] is True
    assert result["per_mlp"][0]["local_comparison_status"] == "rpc_passed_local_failed"
    assert "local_mse_relative_delta" not in result["per_mlp"][0]


def test_limits_cannot_silently_change(monkeypatch, tmp_path, existing_evidence):
    evidence = deepcopy(existing_evidence[0])
    evidence["manifest"]["limits"]["predict_seconds"] = 121
    with pytest.raises(ValueError, match="limit mismatch"):
        score(monkeypatch, tmp_path, evidence)
