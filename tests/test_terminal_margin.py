"""Terminal-stop margin is SOFT, only after all backup hard checks."""
import csv
import pytest

from lidar_sim.model import State, VehicleConfig
from lidar_sim.world import World, MovingObstacle
from lidar_sim.terminal import terminal_stop_margin
from lidar_sim.planner import PlannerConfig, choose_action
from lidar_sim.experiment import RunConfig, run, save_result
from lidar_sim.evaluation import paired_terminal_margin_effects


def test_future_target_arrival_reduces_stop_margin():
    car = VehicleConfig()
    stopped = State(0, 0, 0, 0)
    approaching = World((MovingObstacle("ped", 0, -4, 0, 1, .2),))
    far = World((MovingObstacle("far", 40, 0, 0, 0, .2),))
    a = terminal_stop_margin(stopped, approaching, car, hold_s=6, dt=.1,
                             padding=.12, desired_m=1.2, weight=3)
    b = terminal_stop_margin(stopped, far, car, hold_s=6, dt=.1,
                             padding=.12, desired_m=1.2, weight=3)
    assert a.minimum_clearance_m < b.minimum_clearance_m
    assert a.penalty > b.penalty
    assert b.penalty == 0


def test_no_terminal_preference_is_identical_to_old_strict_planner():
    car = VehicleConfig()
    state = State(0, 0, 0, 1.5)
    world = World((MovingObstacle("ped", 13, -4.4, 0, .58),))
    cfg = PlannerConfig(terminal_margin_weight=0)
    old = choose_action(state, world, car, cfg, mode="braking_hold")
    new = choose_action(state, world, car, cfg, mode="terminal_margin")
    assert old.command == new.command
    assert old.feasible_count == new.feasible_count
    assert old.rejected_by_backup == new.rejected_by_backup
    assert old.score == pytest.approx(new.score)
    assert new.cost_parts["cost_terminal_margin"] == 0


def test_soft_preference_cannot_override_hard_collision_constraint():
    car = VehicleConfig()
    state = State(0, 0, 0, 1.5)
    world = World((MovingObstacle("overlap", 0, 0, 0, 0, .8),))
    cfg = PlannerConfig(terminal_margin_weight=1e6)
    old = choose_action(state, world, car, cfg, mode="braking_hold")
    new = choose_action(state, world, car, cfg, mode="terminal_margin")
    assert old.emergency and new.emergency
    assert old.fallback_strategy == new.fallback_strategy
    assert new.feasible_count == 0


def test_selected_score_breakdown_includes_terminal_preference():
    car = VehicleConfig()
    state = State(0, 0, 0, 1.5)
    plan = choose_action(state, World(()), car, PlannerConfig(), mode="terminal_margin")
    assert not plan.unsafe_fallback
    assert "cost_terminal_margin" in plan.cost_parts
    chosen = [r for r in plan.diagnostics if r["selected"]]
    assert len(chosen) == 1
    assert "terminal_min_clearance_m" in chosen[0]
    assert plan.cost_parts["total_cost"] == pytest.approx(plan.score)


def test_candidate_csv_handles_optional_margin_keys(tmp_path):
    car = VehicleConfig()
    cfg = PlannerConfig()
    state = State(0, 0, 0, 1.5)
    world = World((MovingObstacle("ped", 13, -4.4, 0, .58),))
    plan = choose_action(state, world, car, cfg, mode="terminal_margin")
    outcome = run("empty", mode="terminal_margin",
                  config=RunConfig(max_time=.3))
    outcome.decision_snapshot = {
        "time": 0, "x": 0, "y": 0, "yaw": 0, "speed": 1.5,
        "mode": "terminal_margin", "rows": plan.diagnostics}
    save_result(outcome, tmp_path)
    p = tmp_path / "empty_terminal_margin_0_decision_candidates.csv"
    with p.open(newline="", encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    assert len(records) == 25
    assert "cost_terminal_margin" in records[0]


def test_paired_terminal_margin_does_not_hide_noncollision_failure():
    config = dict(case_id="two_targets__nominal__0", scenario="two_targets",
                  stress="nominal", seed=0, target_speed_scale=1.0,
                  road_half_width_m=3.25, control_delay_s=.25, braking_mps2=1.6)
    old = dict(config, mode="braking_hold", success=False, collision=True,
               road_exit=False, timeout=False, duration_s=19.0)
    new = dict(config, mode="terminal_margin", success=False, collision=False,
               road_exit=False, timeout=True, duration_s=30.0)
    pair = paired_terminal_margin_effects([old, new])[0]
    assert pair["collision_change"] == -1
    assert pair["timeout_change"] == 1
    assert pair["success_change"] == 0
    assert pair["both_success_time_change_s"] == ""
