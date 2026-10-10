"""Prevent invalid network/config comparisons from being presented as parity."""
from copy import deepcopy

import pytest

from benchmark.phase2 import compare


def report():
    return {"run_config": {"flop_budget": 2**41, "seed": 0},
            "results": {"n_failed_mlps": 0, "per_mlp": [
                {"mlp_name": "mini_0", "seed": 42, "final_layer_mse": 2e-8,
                 "adjusted_final_layer_score": 5e-9, "flops_used": 5e11}]}}


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
