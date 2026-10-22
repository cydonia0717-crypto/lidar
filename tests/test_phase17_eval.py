"""Small deterministic checks for phase17 evaluation command."""
from pathlib import Path

import pytest

from lidar_sim.evaluation import TrialSpec
from lidar_sim.phase17_eval import parse_policies, execute


def test_parse_policies():
    assert parse_policies("none,fixed:80,fixed:160,measured") == [
        ("none", 0.0), ("fixed", 80.0), ("fixed", 160.0),
        ("measured", 0.0)]
    for invalid in ("", "fixed:-1", "fixed:nan", "not_a_policy",
                    "none,none"):
        with pytest.raises(ValueError):
            parse_policies(invalid)


def test_phase17_short_eval_persists_full_trace(tmp_path):
    result = execute(
        TrialSpec("empty", 0, "nominal"),
        mode="two_stage_bodyframe", policy="fixed", fixed_ms=200,
        max_time=.4, output=tmp_path)
    assert result["compute_policy"] == "fixed"
    assert result["fixed_compute_ms"] == 200
    assert result["missed_control_ticks"] >= 1
    assert result["submitted_commands"] == result["n_plans"]
    assert 0 <= result["over_100ms_fraction"] <= 1
    assert list(tmp_path.rglob("*_trace.csv"))
    assert list(tmp_path.rglob("*_metrics.json"))
