"""Rear-axle reference geometry and honest goal-completion accounting."""
import csv
import math
from dataclasses import replace

import pytest

from lidar_sim.model import State, VehicleConfig
from lidar_sim.world import World, MovingObstacle
from lidar_sim.goal import goal_metrics
from lidar_sim.planner import PlannerConfig, choose_action, collision_free
from lidar_sim.evaluation import (TrialSpec, evaluate_cases, paired_bodyframe_effects,
                                  save_study)
from lidar_sim.experiment import RunConfig, run


def test_rear_axle_transform_uses_vehicle_body_center():
    car = replace(VehicleConfig(), road_footprint="rearaxle_rectangle")
    s = State(10, 1, math.pi/2, 0)
    x, y = car.body_center(s)
    assert x == pytest.approx(10)
    assert y == pytest.approx(1.55)
    assert replace(car, road_footprint="rectangle").body_center(s) == (10, 1)


def test_true_body_road_boundary_accounts_for_offset():
    road = World((), road_half_width=3.25)
    old = replace(VehicleConfig(), road_footprint="rectangle")
    true = replace(old, road_footprint="rearaxle_rectangle")
    s = State(10, 1.5, math.pi/2, 0)
    assert road.road_clearance(s, old) == pytest.approx(3.25 - 1.5 - .95)
    assert road.road_clearance(s, true) == pytest.approx(3.25 - 2.05 - .95)
    assert road.road_clearance(s, true) < road.road_clearance(s, old)


def test_target_collision_center_also_uses_rear_axle_offset():
    true = replace(VehicleConfig(), road_footprint="rearaxle_rectangle")
    state = State(0, 0, 0, 0)
    world = World((MovingObstacle("person", 1.7, 0, 0, 0, .35),))
    assert world.obstacle_clearance(state, true) < world.obstacle_clearance(
        state, replace(true, road_footprint="rectangle"))
    assert not collision_free([state], world, true, padding=.12)


def test_goal_requires_physical_lateral_alignment_not_only_x():
    w = World(())
    car = replace(VehicleConfig(), road_footprint="rearaxle_rectangle")
    x_threshold = 27.4
    bad = goal_metrics(State(x_threshold, 1.2, math.radians(35), 1.5), w, car)
    assert bad["historical_x_line_met"]
    assert bad["center_x_line_met"]
    assert not bad["pose_gate_met"]
    assert not bad["lateral_met"]
    assert not bad["heading_met"]
    good = goal_metrics(State(27.0, 0, 0, 1.5), w, car)
    assert not good["historical_x_line_met"]
    assert good["pose_gate_met"]  # body center at 27.55m
    assert goal_metrics(State(27., 0, 0, 1.5), w, car,
                        speed_limit_mps=.5)["pose_gate_met"] is False


def test_goal_heading_wrap_near_positive_pi():
    w = World(())
    car = VehicleConfig()
    bad = goal_metrics(State(27.7, 0, math.pi, .1), w, car)
    assert bad["historical_x_line_met"] and not bad["heading_met"]


def test_rear_axle_offset_invalid_and_goal_invalid_fail_loudly():
    car=replace(VehicleConfig(),road_footprint="rearaxle_rectangle",
                rear_axle_to_body_center_m=100)
    with pytest.raises(ValueError,match="offset"):
        car.body_center(State(0,0,0,0))
    with pytest.raises(ValueError,match="tolerances"):
        goal_metrics(State(0,0,0,0),World(()),VehicleConfig(),
                     lateral_tolerance_m=-.5)
    with pytest.raises(ValueError,match="unknown goal criterion"):
        run("empty",config=RunConfig(max_time=.2,goal_criterion="easy"))
    with pytest.raises(ValueError,match="unknown physical"):
        run("empty",config=RunConfig(max_time=.2,physical_road_footprint="fake"))


def test_pose_gate_in_trace_and_metrics_are_consistent():
    config=RunConfig(max_time=.3,goal_criterion="pose_gate",
                     physical_road_footprint="rearaxle_rectangle")
    result=run("empty",mode="two_stage_bodyframe",config=config)
    assert result.goal_criterion=="pose_gate"
    assert "pose_gate_met" in result.trace[0]
    assert "body_center_x" in result.trace[0]
    assert "historical_x_line_met" in result.trace[0]
    assert result.reached_pose_gate is False
    assert result.reached_historical_x_line is False


def test_same_physical_truth_and_pose_gate_required_for_paired_ablation(tmp_path):
    records=evaluate_cases([TrialSpec("empty", 0,"nominal")],
        ("two_stage_rectroad","two_stage_bodyframe"),
        max_time=.25,physical_road_footprint="rearaxle_rectangle",
        goal_criterion="pose_gate")
    assert len(records)==2
    assert {r["physical_road_footprint"] for r in records}=={"rearaxle_rectangle"}
    assert {r["goal_criterion"] for r in records}=={"pose_gate"}
    assert len(paired_bodyframe_effects(records))==1
    files=save_study(records,tmp_path)
    with files["paired_bodyframe"].open(newline="",encoding="utf8") as fh:
        assert len(list(csv.DictReader(fh)))==1
    with pytest.raises(ValueError,match="unpaired physical"):
        paired_bodyframe_effects([records[0],
            dict(records[1],goal_lateral_tolerance_m=1.5)])
    with pytest.raises(ValueError,match="common rear axle"):
        paired_bodyframe_effects([dict(r,physical_road_footprint="rectangle")
                                  for r in records])


def test_existing_legacy_mode_remains_accessible():
    old=run("empty",config=RunConfig(max_time=.3))
    assert old.goal_criterion=="x_gate"
    assert not old.collision
    old_planner=choose_action(State(0,0,0,1.5),World(()),
        VehicleConfig(),PlannerConfig(),mode="two_stage_rectroad")
    assert old_planner.feasible_count>0
