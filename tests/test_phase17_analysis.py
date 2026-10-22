"""Phase17 paired comparisons reject missing or duplicate policies."""
import pytest

from lidar_sim.phase17_analysis import pair_rows


def sample(policy, **changes):
    row = {
        "case_id": "two_targets__weak_brake__1",
        "mode": "two_stage_viability",
        "compute_policy": policy,
        "fixed_compute_ms": "0.0",
        "compute_scale": "1.0",
        "success": "True", "collision": "False", "timeout": "False",
        "road_exit": "False", "duration_s": "16.6",
        "min_clearance_m": ".32",
        "mean_planning_ms": "54.1",
        "p95_planning_ms": "80",
        "p99_planning_ms": "110",
        "over_100ms_fraction": ".01",
        "missed_control_ticks": "1",
        "submitted_commands": "165",
    }
    row.update(changes)
    return row


def test_phase17_paired_outcomes_record_latency_changes():
    rows = pair_rows([sample("none"), sample(
        "measured", success="False", collision="True",
        missed_control_ticks="3")])
    assert len(rows) == 1
    assert rows[0]["none_success"] == "True"
    assert rows[0]["measured_success"] == "False"
    assert rows[0]["measured_collision"] == "True"
    assert rows[0]["measured_missed_control_ticks"] == "3"


@pytest.mark.parametrize("rows", [
    [sample("none")],
    [sample("measured")],
    [sample("none"), sample("none"), sample("measured")],
    [sample("none"), sample("measured", compute_scale="2.0")],
])
def test_phase17_paired_outcomes_fail_on_unmatched_inputs(rows):
    with pytest.raises(ValueError):
        pair_rows(rows)
