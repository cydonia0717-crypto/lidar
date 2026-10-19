"""Strict feasibility rejection phases are inspectable and non-invasive."""
import csv

import pytest

from lidar_sim.model import State, VehicleConfig
from lidar_sim.world import World, MovingObstacle
from lidar_sim.planner import PlannerConfig, choose_action
from lidar_sim.two_stage import first_violation_kind, blank_rejections
from lidar_sim.experiment import RunConfig, run, save_result
from lidar_sim.rejection_diagnostics import rejection_windows, PHASES, CAUSES


def test_first_violation_distinguishes_road_from_dynamic_obstacle():
    car = VehicleConfig()
    state = State(0,0,0,0,time=1)
    road = World((), road_half_width=.5)
    assert first_violation_kind([state], road, car, 0) == "road"
    obstacle = World((MovingObstacle("blocked", 0,0,0,0,.4),))
    assert first_violation_kind([state], obstacle, car, .12) == "obstacle"
    with pytest.raises(ValueError, match="no violation"):
        first_violation_kind([state], World(()), car, 0)


def test_overlap_prefix_returns_explicit_reason_without_search():
    cfg = PlannerConfig()
    car = VehicleConfig()
    world = World((MovingObstacle("blocked", 0,0,0,0,1),))
    plan = choose_action(State(0,0,0,0), world, car, cfg, mode="two_stage")
    assert plan.emergency
    assert plan.rejection_counts["reject_prefix_obstacle"] == 1
    assert plan.candidate_count == 0
    assert sum(plan.rejection_counts.values()) == 1


def test_rejection_reason_instrumentation_does_not_rewrite_viability():
    cfg = PlannerConfig()
    car = VehicleConfig()
    plan = choose_action(State(0,0,0,1.5), World(()), car, cfg, mode="two_stage")
    assert plan.feasible_count > 0
    assert len(plan.rejection_counts) == len(PHASES) * len(CAUSES)
    assert all(isinstance(v, int) and v >= 0 for v in plan.rejection_counts.values())
    assert plan.rejected_by_backup == sum(
        plan.rejection_counts[f"reject_{phase}_{cause}"]
        for phase in ("brake", "hold") for cause in CAUSES)
    assert all(k in blank_rejections() for k in plan.rejection_counts)


def test_source_rejections_are_exported_to_real_trace_and_csv(tmp_path):
    result = run("empty", mode="two_stage", config=RunConfig(max_time=.3))
    live = result.trace[0]
    assert "moving_first_action_feasible" in live
    assert "max_feasible_first_speed" in live
    assert "reject_first_obstacle" in live
    assert live["moving_first_action_feasible"] == 1
    _, trace_csv = save_result(result, tmp_path)
    with trace_csv.open(newline="", encoding="utf-8") as fh:
        stored = list(csv.DictReader(fh))
    assert "reject_hold_obstacle" in stored[0]
    assert "feasible_first_actions" in stored[0]


def test_aggregation_distinguishes_stage_counts_from_forward_options():
    record = {
        "time": 20, "x": 14.0, "command_speed": 0.0,
        "feasible_pairs": 5, "moving_first_action_feasible": 0,
        "reject_first_obstacle": 8, "reject_second_obstacle": 5,
        "reject_brake_road": 2, "reject_hold_obstacle": 9,
        "emergency": 0,
    }
    window = rejection_windows([record], span_s=5)[0]
    assert window["moving_first_available_fraction"] == 0
    assert window["mean_feasible_pairs"] == 5
    assert window["reject_first_obstacle"] == 8
    assert window["reject_second_obstacle"] == 5
    assert window["reject_brake_road"] == 2
    assert window["reject_hold_obstacle"] == 9
    assert window["emergency_fraction"] == 0
