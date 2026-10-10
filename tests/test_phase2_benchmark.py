"""Prevent invalid network/config comparisons from being presented as parity."""

from copy import deepcopy

import pytest

from benchmark.phase2 import compare


def report():
    return {
        "run_config": {"flop_budget": 2**41, "seed": 0},
        "results": {
            "n_failed_mlps": 0,
            "per_mlp": [
                {
                    "mlp_name": "mini_0",
                    "seed": 42,
                    "final_layer_mse": 2e-8,
                    "adjusted_final_layer_score": 5e-9,
                    "flops_used": 5e11,
                }
            ],
        },
    }


def test_different_network_rejected():
    a = report()
    b = deepcopy(a)
    b["results"]["per_mlp"][0]["seed"] = 43
    with pytest.raises(ValueError, match="Network order"):
        compare(a, b)


def test_different_budget_rejected():
    a = report()
    b = deepcopy(a)
    b["run_config"]["flop_budget"] = 272000000000
    with pytest.raises(ValueError, match="flop_budget"):
        compare(a, b)


def test_paired_difference():
    a = report()
    b = deepcopy(a)
    b["results"]["per_mlp"][0]["final_layer_mse"] = 1e-8
    result = compare(a, b)
    assert result["per_mlp"][0]["final_layer_mse"]["relative_delta"] == pytest.approx(-0.5)


def test_online_pending_is_not_a_score_comparison():
    from benchmark.summarize import summarize

    local = report()
    local["results"].update(
        adjusted_final_layer_score=5e-9, final_layer_mse=2e-8, mean_compute_utilization=0.25
    )
    online = {"response": {"grading_status_cd": "submitted", "score": None}}
    runners = {"same_flops": True, "max_relative_mse_delta": 0, "left_failed": 0, "right_failed": 0}
    submission = {"submission_id": 1, "estimator_sha256": "hash", "artifact_sha256": "hash"}
    result = summarize(local, online, runners, submission)
    assert result["online_vs_local_aggregate_relative_delta"] is None
    assert result["online_same_network_parity_verified"] is False


def test_online_graded_aggregate_is_not_same_network_parity():
    from benchmark.summarize import summarize

    local = report()
    local["results"].update(
        adjusted_final_layer_score=5e-9, final_layer_mse=2e-8, mean_compute_utilization=0.25
    )
    online = {"response": {"grading_status_cd": "graded", "score": 6e-9}}
    runners = {"same_flops": True, "max_relative_mse_delta": 0, "left_failed": 0, "right_failed": 0}
    submission = {"submission_id": 1, "estimator_sha256": "hash", "artifact_sha256": "hash"}
    result = summarize(local, online, runners, submission)
    assert result["online_vs_local_aggregate_relative_delta"] == pytest.approx(0.2)
    assert result["online_same_network_parity_verified"] is False
