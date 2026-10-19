"""Regression tests for the risk-ranked, explicitly unverified recovery mode."""
import csv
import math

import pytest

from lidar_sim.model import Command, State, VehicleConfig
from lidar_sim.planner import PlannerConfig, choose_action
from lidar_sim.recovery import Risk, rank_rescue, risk_of_path
from lidar_sim.world import MovingObstacle, World
from lidar_sim.experiment import RunConfig, run, save_result


def test_risk_exposure_uses_actual_time_and_earliest_collision():
    car = VehicleConfig()
    world = World((MovingObstacle("static", 0, 0, 0, 0, .2),), road_half_width=4)
    path = [State(0, 0, 0, 0, time=0),
            State(0, 0, 0, 0, time=.2),
            State(0, 0, 0, 0, time=.5)]
    risk = risk_of_path(path, world, car, padding=.1)
    assert risk.minimum_clearance_m < 0
    assert risk.violation_exposure > 0
    assert risk.first_violation_after_s == 0
    with pytest.raises(ValueError):
        risk_of_path(path[:1] + path[:1], world, car)


def test_lexicographic_recovery_uses_nominal_hard_constraint_first():
    better_cost_but_collision = rank_rescue(
        nominal_collision_free=False, risk=Risk(-.01, .001, 2.0),
        nominal_cost=-999)
    safe_nominal = rank_rescue(
        nominal_collision_free=True, risk=Risk(-.2, .5, 1.0),
        nominal_cost=100)
    assert safe_nominal < better_cost_but_collision


def test_no_recovery_action_can_ignore_imminent_collision():
    car = VehicleConfig()
    start = State(0, 0, 0, 0)
    world = World((MovingObstacle("already_touching", 0, 0, 0, 0, 1),))
    plan = choose_action(start, world, car, PlannerConfig(), mode="recovery")
    assert plan.emergency
    assert plan.unsafe_fallback
    assert plan.rescue_immediate_feasible_count == 0
    assert plan.fallback_strategy == "emergency_brake"


def test_when_safe_candidate_exists_recovery_matches_strict_hold():
    car = VehicleConfig()
    state = State(0, 0, 0, 1.5)
    cfg = PlannerConfig()
    old = choose_action(state, World(()), car, cfg, mode="braking_hold")
    new = choose_action(state, World(()), car, cfg, mode="recovery")
    assert old.command == new.command
    assert old.feasible_count == new.feasible_count
    assert not new.unsafe_fallback
    assert new.fallback_strategy == "none"


def test_recovery_logs_immediate_safe_actions_even_when_long_backup_fails():
    car = VehicleConfig()
    state = State(0, 0, 0, 0)
    world = World((MovingObstacle("crossing", 0, -4, 0, 1, .2),))
    cfg = PlannerConfig()
    strict = choose_action(state, world, car, cfg, mode="braking_hold")
    rescue = choose_action(state, world, car, cfg, mode="recovery")
    assert rescue.rescue_immediate_feasible_count > 0
    assert rescue.backup_checked
    assert rescue.rejected_by_backup >= strict.rejected_by_backup
    assert rescue.fallback_strategy in ("none", "risk_ranked_no_backup")
    if rescue.fallback_strategy == "risk_ranked_no_backup":
        assert rescue.unsafe_fallback
        assert rescue.fallback_risk_exposure is not None
        selected = [x for x in rescue.diagnostics if x["selected"]]
        assert len(selected) == 1
        assert selected[0]["immediate_safe"]


def test_recovery_trace_and_candidate_csv_allow_sparse_extra_fields(tmp_path):
    # Even the normal fallback must remain inspectable and not generate
    # malformed CSV when risk fields exist only for some candidates.
    outcome = run("empty", 0, "recovery", config=RunConfig(max_time=.5))
    assert outcome.risk_ranked_fallback_steps == 0
    metrics, trace = save_result(outcome, tmp_path)
    with trace.open(newline="", encoding="utf-8") as fd:
        rows = list(csv.DictReader(fd))
    assert "fallback_risk_exposure" in rows[0]
    assert "rescue_immediate_feasible_count" in rows[0]
