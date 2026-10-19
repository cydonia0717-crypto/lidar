"""A stopped ego can still be hit by moving traffic after braking ends."""
import pytest

from lidar_sim.model import Command, State, VehicleConfig
from lidar_sim.planner import PlannerConfig, choose_action, collision_free, rollout
from lidar_sim.safety import (CommandDelayLine, backup_segments,
                              braking_then_hold, stationary_hold)
from lidar_sim.world import MovingObstacle, World


def test_stationary_hold_detects_late_arrival_after_physical_braking():
    car = VehicleConfig()
    initial = State(0, 0, 0, 1.0)
    line = CommandDelayLine(Command(1.0, 0.0), delay_s=0.0)
    cfg = PlannerConfig(control_delay_s=0.0, candidate_execute_s=.4,
                        stationary_hold_s=6.0)
    world = World((MovingObstacle("late_pedestrian", .28, -5.0, 0, 1, .20),))
    command = Command(0.0, 0.0)
    nominal = rollout(initial, command, car, cfg.horizon, cfg.sample_dt)
    assert collision_free(nominal, world, car, padding=cfg.prediction_padding)
    segments = backup_segments(initial, command, line, car,
                               cfg.candidate_execute_s, cfg.safety_dt)
    assert segments.brake[-1].speed == pytest.approx(0)
    assert collision_free(segments.complete, world, car,
                          padding=cfg.prediction_padding)
    held = braking_then_hold(segments, cfg.stationary_hold_s, cfg.safety_dt)
    assert held[-1].time == pytest.approx(
        segments.brake[-1].time + cfg.stationary_hold_s)
    assert all(p.speed == 0 for p in held[len(segments.complete):])
    assert not collision_free(held, world, car,
                              padding=cfg.prediction_padding)


def test_braking_hold_can_reject_more_actions_than_legacy_braking():
    car = VehicleConfig()
    initial = State(0, 0, 0, 1)
    cfg = PlannerConfig(control_delay_s=0,
                        candidate_execute_s=.4, stationary_hold_s=6.0)
    world = World((MovingObstacle("late_pedestrian", .28, -5., 0, 1, .20),))
    old = choose_action(initial, world, car, cfg, mode="braking")
    new = choose_action(initial, world, car, cfg, mode="braking_hold")
    assert old.backup_checked and new.backup_checked
    assert new.rejected_by_backup >= old.rejected_by_backup
    assert any(d["status"] == "backup_rejected" for d in new.diagnostics)


def test_zero_hold_is_identical_to_legacy_braking():
    car = VehicleConfig()
    initial = State(0, 0, 0, 1.5)
    cfg = PlannerConfig(stationary_hold_s=0.0)
    world = World((MovingObstacle("ped", 10, -2, 0, .6, .35),))
    old = choose_action(initial, world, car, cfg, mode="braking")
    new = choose_action(initial, world, car, cfg, mode="braking_hold")
    assert old.command == new.command
    assert old.feasible_count == new.feasible_count
    assert old.rejected_by_backup == new.rejected_by_backup


def test_stopped_hold_path_preserves_position():
    state = State(4, -.5, .2, 0, -.2, 2.0)
    hold = stationary_hold(state, 2.5, .1)
    assert hold[-1].time == pytest.approx(4.5)
    assert all(p.x == 4 and p.y == -.5 and p.speed == 0 for p in hold)
    with pytest.raises(ValueError):
        stationary_hold(State(0, 0, 0, 1), 1, .1)
    with pytest.raises(ValueError):
        stationary_hold(state, -1, .1)
