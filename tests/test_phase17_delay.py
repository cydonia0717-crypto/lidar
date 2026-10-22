"""Phase17 CPU-lag / single-worker controller timing invariants."""
from dataclasses import replace

import pytest

from lidar_sim.experiment import RunConfig, run
from lidar_sim.phase16_equivalence import PHYSICAL_KEYS


def test_zero_extra_compute_delay_is_physical_replay_equivalent():
    base = RunConfig(max_time=.6, physical_road_footprint="rearaxle_rectangle",
                     goal_criterion="pose_gate", physical_body_offset_m=.55)
    zero = replace(base, compute_delay_mode="fixed", compute_fixed_ms=0)
    old = run("empty", mode="two_stage_viability", config=base)
    new = run("empty", mode="two_stage_viability", config=zero)
    assert len(old.trace) == len(new.trace)
    for a, b in zip(old.trace, new.trace):
        for field in PHYSICAL_KEYS:
            assert a.get(field) == b.get(field), field
    assert new.missed_control_ticks == 0
    assert new.submitted_commands == len(new.trace) - 1


def test_fixed_200ms_compute_queues_command_and_skips_busy_ticks():
    base = RunConfig(max_time=.6, compute_delay_mode="fixed",
                     compute_fixed_ms=200)
    result = run("empty", mode="two_stage_bodyframe", config=base)
    assert result.missed_control_ticks >= 2
    assert result.submitted_commands < len(result.trace) - 1
    decision_rows = [r for r in result.trace if r.get("compute_busy") == 0]
    skipped_rows = [r for r in result.trace if r.get("compute_busy") == 1]
    assert decision_rows and skipped_rows
    assert all(r["compute_submit_delay_ms"] == pytest.approx(200)
               for r in decision_rows)
    assert all(r["planning_ms"] == "" for r in skipped_rows)
    assert all(r["command_speed"] == "" for r in skipped_rows)
    # The first command is issued at t=0.2 and actuates at t=0.45;
    # the plant must still be using its original target at 0.4.
    before_activation = next(r for r in result.trace
                             if abs(r["time"] - .4) < 1e-7)
    assert before_activation["applied_speed"] == pytest.approx(1.5)


def test_measured_compute_delay_matches_logged_wall_time():
    rcfg = RunConfig(max_time=.3, compute_delay_mode="measured")
    result = run("empty", mode="two_stage_bodyframe", config=rcfg)
    decisions = [r for r in result.trace if r.get("compute_busy") == 0]
    assert decisions
    assert all(r["compute_submit_delay_ms"] == pytest.approx(r["planning_ms"])
               for r in decisions)
    assert result.submitted_commands > 0


@pytest.mark.parametrize("cfg", [
    RunConfig(compute_delay_mode="unknown"),
    RunConfig(compute_delay_mode="fixed", compute_fixed_ms=-10),
    RunConfig(compute_delay_mode="fixed", compute_delay_scale=0),
    RunConfig(compute_delay_mode="fixed", compute_delay_scale=float("inf")),
])
def test_invalid_compute_delay_configuration_rejected(cfg):
    with pytest.raises(ValueError):
        run("empty", mode="two_stage_bodyframe", config=cfg)
