"""Regression tests for explicit own-vehicle trajectory decision layer."""
import csv
import json
import math

import pytest

from lidar_sim.decision import trajectory_cost_breakdown
from lidar_sim.experiment import run, save_result
from lidar_sim.model import Command, State, VehicleConfig
from lidar_sim.planner import PlannerConfig, choose_action, rollout
from lidar_sim.world import MovingObstacle, World


def test_weighted_score_is_exact_sum_of_reported_parts():
    car = VehicleConfig()
    cfg = PlannerConfig()
    start = State(0, 0, 0, 1.0)
    command = Command(1.8, .21)
    path = rollout(start, command, car, 1.2, .1)
    cost = trajectory_cost_breakdown(path, command, start, World(()), car, cfg)
    record = cost.as_record()
    assert record["total_cost"] == pytest.approx(sum(
        value for key, value in record.items() if key.startswith("cost_")))
    assert record["cost_progress"] < 0
    assert record["cost_route"] >= 0
    assert record["cost_speed_change"] > 0


def test_proximity_penalty_increases_for_near_obstacle():
    car = VehicleConfig()
    cfg = PlannerConfig()
    start = State(0, 0, 0, 1)
    command = Command(1.0, 0)
    near = [State(0, 0, 0, 1, time=0), State(1, 0, 0, 1, time=1)]
    far = [State(0, 1.5, 0, 1, time=0), State(1, 1.5, 0, 1, time=1)]
    world = World((MovingObstacle("static", 1, 0, 0, 0, .2),), road_half_width=5)
    cost_near = trajectory_cost_breakdown(near, command, start, world, car, cfg)
    cost_far = trajectory_cost_breakdown(far, command, start, world, car, cfg)
    assert cost_near.clearance > cost_far.clearance


def test_hard_constraints_cannot_be_overridden_by_soft_reward():
    car = VehicleConfig()
    cfg = PlannerConfig(progress_weight=10000, route_weight=0, clearance_weight=0)
    world = World((MovingObstacle("blocking", 2, 0, 0, 0, 1),))
    plan = choose_action(State(0, 0, 0, 1), world, car, cfg)
    assert plan.candidate_count == 25
    assert len(plan.diagnostics) == 25
    assert any(r["status"] == "nominal_rejected" for r in plan.diagnostics)
    assert not any(r["selected"] and r["status"] == "nominal_rejected"
                   for r in plan.diagnostics)


def test_stationary_obstacle_has_same_padding_for_reactive_and_predictive():
    car = VehicleConfig()
    cfg = PlannerConfig()
    world = World((MovingObstacle("static", 9, 0, 0, 0, .4),))
    start = State(0, 0, 0, 1.5)
    reactive = choose_action(start, world, car, cfg, mode="reactive")
    predictive = choose_action(start, world, car, cfg, mode="predictive")
    assert reactive.feasible_count == predictive.feasible_count
    assert reactive.command == predictive.command


def test_chosen_score_and_diagnostics_agree():
    plan = choose_action(State(0, 0, 0, 1.5), World(()),
                         VehicleConfig(), PlannerConfig(), mode="braking")
    chosen = [r for r in plan.diagnostics if r["selected"]]
    assert len(chosen) == 1
    assert chosen[0]["total_cost"] == pytest.approx(plan.score)
    assert chosen[0]["status"] == "selected"
    assert plan.cost_parts["total_cost"] == pytest.approx(plan.score)


def test_snapshot_export_has_all_25_candidates(tmp_path):
    result = run("crossing", seed=0, mode="braking")
    metrics, path = save_result(result, tmp_path)
    assert metrics.exists() and path.exists()
    assert result.decision_snapshot is not None
    assert len(result.decision_snapshot["rows"]) == 25
    snap = tmp_path / "crossing_braking_0_decision_candidates.csv"
    assert snap.exists()
    with snap.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 25
    assert "cost_clearance" in records[0]
    assert "status" in records[0]
    with path.open(newline="", encoding="utf-8") as f:
        trace = list(csv.DictReader(f))
    assert "total_cost" in trace[0]
    assert "cost_route" in trace[0]
    assert "decision_snapshot" not in json.loads(metrics.read_text(encoding="utf-8"))


def test_decision_figure_from_logged_candidates(tmp_path):
    from lidar_sim.visualize import save_decision_figure
    result = run("crossing", seed=0, mode="predictive")
    out = save_decision_figure(result, tmp_path / "decision.png")
    assert out.stat().st_size > 5_000


def test_weight_setting_changes_reported_component():
    car = VehicleConfig()
    start = State(0, 0, 0, 1)
    path = rollout(start, Command(1.8, 0.2), car, 1.0, .1)
    world = World(())
    low = trajectory_cost_breakdown(path, Command(1.8, 0.2), start,
                                    world, car, PlannerConfig(heading_weight=0))
    high = trajectory_cost_breakdown(path, Command(1.8, 0.2), start,
                                     world, car, PlannerConfig(heading_weight=5))
    assert low.heading == 0
    assert high.heading > 0
    assert high.total > low.total
