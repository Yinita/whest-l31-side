"""Prevent duplicate coverage, protocol mixing, and hidden failures in public summaries."""

import json
from pathlib import Path

import pytest

from benchmark.aggregate_strict_public import aggregate

ROOT = Path(__file__).resolve().parents[1]
COHORTS = [ROOT / f"results/phase2/sprint_public_v37b_h18_r0{i}_strict_shard0{i-6}_score.json" for i in (8, 9)]


def modified(tmp_path, changes):
    report = json.loads(COHORTS[1].read_text())
    changes(report)
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(report))
    return path


def test_count_and_weight_networks():
    result = aggregate(COHORTS)
    assert result["n_mlps"] == 14
    assert result["n_failed_mlps"] == 0
    assert len({(r["mlp_name"], r["seed"]) for r in result["per_mlp"]}) == 14
    assert result["adjusted_final_layer_score"] == pytest.approx(4.320574279115936e-9)


def test_duplicate_cohort_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([COHORTS[0], COHORTS[0]])


def test_budget_mismatch_rejected(tmp_path):
    path = modified(tmp_path, lambda r: r["limits"].update(flop_budget=2**42))
    with pytest.raises(ValueError, match="Protocol mismatch: limits"):
        aggregate([COHORTS[0], path])


def test_failure_is_included(tmp_path):
    def mark_failure(report):
        row = report["per_mlp"][0]
        row.update(limits_passed=False, final_layer_mse=1.0, adjusted_final_layer_score=1.0)

    path = modified(tmp_path, mark_failure)
    result = aggregate([COHORTS[0], path])
    assert result["n_mlps"] == 14
    assert result["n_failed_mlps"] == 1
    assert result["adjusted_final_layer_score"] > 1/14
