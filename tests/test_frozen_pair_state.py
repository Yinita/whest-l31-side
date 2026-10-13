"""Workflow completion must clear stale running state without touching another owner."""

import json

from benchmark.run_frozen_public_pair import synchronize_state


def test_complete_clears_running_preserves_research(tmp_path):
    workflow = tmp_path/"workflow.json"
    state = tmp_path/"state.json"
    state.write_text(json.dumps({"active_experiment": {"workflow": str(workflow), "status": "running", "phase": "inputs"},
                                 "stage": 1, "current_target_score": 5.779e-10}))
    report = {"status": "completed", "current_phase": None, "finished_at_utc": "2026-10-13T06:11:40Z",
              "comparison": "paired.json"}
    assert synchronize_state(state, workflow, report)
    result = json.loads(state.read_text())
    assert result["active_experiment"] is None
    assert result["latest_completed_experiment"]["status"] == "completed"
    assert result["stage"] == 1 and result["current_target_score"] == 5.779e-10


def test_other_workflow_is_never_overwritten(tmp_path):
    state = tmp_path/"state.json"
    state.write_text(json.dumps({"active_experiment": {"workflow": str(tmp_path/"other.json")}}))
    original = state.read_bytes()
    assert not synchronize_state(state, tmp_path/"workflow.json", {"status": "running", "current_phase": "inputs"})
    assert state.read_bytes() == original


def test_failure_remains_visible_and_updates_phase(tmp_path):
    workflow = tmp_path/"workflow.json"
    state = tmp_path/"state.json"
    state.write_text(json.dumps({"active_experiment": {"workflow": str(workflow), "status": "running"}}))
    assert synchronize_state(state, workflow, {"status": "failed", "current_phase": "V29_predict", "error": "partial output"})
    active = json.loads(state.read_text())["active_experiment"]
    assert active["status"] == "failed" and active["phase"] == "V29_predict"
    assert active["error"] == "partial output"
