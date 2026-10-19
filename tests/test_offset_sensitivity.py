"""Experiment calibration offsets are independently controlled and audited."""
import csv
from dataclasses import replace
from math import pi

import pytest

from lidar_sim.evaluation import TrialSpec, evaluate_cases
from lidar_sim.experiment import RunConfig, run
from lidar_sim.model import State, VehicleConfig
from lidar_sim.world import World, MovingObstacle
from lidar_sim.offset_sensitivity import (
    parse_offsets, true_gap, forensic_frames, summarize_windows, grid_study)


def test_validated_parameter_range_and_duplicates():
    assert parse_offsets(".4,.55,.7") == (.4,.55,.7)
    for invalid in ("", ".55,.55", "-.1", "1.0", "abc", "nan", "inf"):
        with pytest.raises(ValueError, match="offset"):
            parse_offsets(invalid)


def test_physical_target_gap_uses_truth_not_assumed():
    world=World((MovingObstacle("ped", 2,0,0,0,.1),))
    state=State(0,0,0,0)
    base=VehicleConfig(road_footprint="rearaxle_rectangle")
    far=true_gap(state,world,replace(base,rear_axle_to_body_center_m=.4))
    near=true_gap(state,world,replace(base,rear_axle_to_body_center_m=.7))
    assert near["physical_obstacle_gap_m"] < far["physical_obstacle_gap_m"]
    assert near["physical_road_gap_m"] == pytest.approx(far["physical_road_gap_m"])


def test_physical_and_planning_offsets_independent_in_trace():
    args=dict(mode="two_stage_bodyframe",scenario_name="empty",
              vehicle=VehicleConfig(rear_axle_to_body_center_m=.4),
              config=RunConfig(max_time=.2,goal_criterion="pose_gate",
                  physical_road_footprint="rearaxle_rectangle",
                  physical_body_offset_m=.7))
    mismatched=run(**args)
    assert mismatched.physical_body_offset_m==pytest.approx(.7)
    assert mismatched.assumed_body_offset_m==pytest.approx(.4)
    assert mismatched.trace[0]["body_center_x"]==pytest.approx(.7)
    matched=run(**(args | {"config":replace(args["config"],
                                              physical_body_offset_m=.4)}))
    assert matched.trace[0]["body_center_x"]==pytest.approx(.4)
    assert matched.assumed_body_offset_m == pytest.approx(.4)


def test_invalid_physical_offset_rejected_before_planning():
    with pytest.raises(ValueError,match="physical body offset"):
        run("empty",config=RunConfig(max_time=.1,
            physical_body_offset_m=-.01))
    with pytest.raises(ValueError,match="physical body offset"):
        run("empty",config=RunConfig(max_time=.1,
            physical_body_offset_m=float("nan")))
    with pytest.raises(ValueError,match="planning body offset"):
        run("empty",vehicle=VehicleConfig(rear_axle_to_body_center_m=1.0),
            config=RunConfig(max_time=.1))


def test_evaluation_records_offset_pair_without_changing_world_seed():
    spec=TrialSpec("empty",0,"nominal")
    rows=evaluate_cases([spec],("two_stage_bodyframe",),
        max_time=.15,goal_criterion="pose_gate",
        physical_road_footprint="rearaxle_rectangle",
        assumed_body_offset_m=.4,physical_body_offset_m=.7)
    assert rows[0]["physical_body_offset_m"]==.7
    assert rows[0]["assumed_body_offset_m"]==.4
    assert rows[0]["goal_criterion"]=="pose_gate"
    with pytest.raises(ValueError,match="invalid rear-axle body offset"):
        evaluate_cases([spec],("two_stage_bodyframe",),
            physical_body_offset_m=1.2)


def test_forensic_frames_preserve_original_and_window_separates_timeout():
    result=run("empty",mode="two_stage_bodyframe",
        config=RunConfig(max_time=.3,physical_road_footprint="rearaxle_rectangle",
                         goal_criterion="pose_gate"))
    original=list(result.trace)
    frames=forensic_frames(result.trace,World(()),
                          replace(VehicleConfig(),road_footprint="rearaxle_rectangle"))
    assert result.trace == original
    assert len(frames)==len(result.trace)
    assert "physical_obstacle_gap_m" in frames[0]
    assert "reject_second_road" in frames[0]
    windows=summarize_windows(frames,window_s=5)
    assert len(windows)==1
    assert "reject_hold_obstacle" in windows[0]


def test_short_smoke_grid_saves_each_result_and_metadata(tmp_path):
    out=tmp_path/"grid"
    rows=grid_study(scenarios=("empty",),stresses=("nominal",),seeds=(0,),
         physical_offsets=(.4,.7),assumed_offsets=(.4,.55),
         max_time=.2,output=out,forensic_seeds=())
    assert len(rows)==4
    with (out/"episodes.csv").open(newline="",encoding="utf8") as fd:
        actual=list(csv.DictReader(fd))
    assert len(actual)==4
    assert {(r["physical_body_offset_m"],r["assumed_body_offset_m"])
             for r in actual} == {(".4",".4"),(".4",".55"),(".7",".4"),(".7",".55")}
    assert (out/"design.json").exists()
    assert (out/"offset_grid_summary.csv").exists()
