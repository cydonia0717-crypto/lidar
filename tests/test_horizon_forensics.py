"""No-policy-change diagnostics for 30s cutoff and stopped cyclist intrusion."""
from dataclasses import replace

import pytest

from lidar_sim.horizon_forensics import (
    parse_horizons, fixed_pose_intrusion, prefix_comparison,
    annotate_trace, windows, study)
from lidar_sim.model import State, VehicleConfig
from lidar_sim.world import World, MovingObstacle
from lidar_sim.experiment import run, RunConfig
from lidar_sim.evaluation import TrialSpec, make_trial


def test_horizons_are_sorted_and_validated():
    assert parse_horizons("40,30,35") == (30.,35.,40.)
    for value in ("", "30", "30,30", "30,NaN", "inf,35", "-1,35", "bad,40"):
        with pytest.raises(ValueError, match="horizons"):
            parse_horizons(value)


def test_parked_intrusion_distinguishes_physical_and_planner_envelopes():
    world=World((MovingObstacle("cyclist",3,0,-.5,0,.2),))
    car=replace(VehicleConfig(), road_footprint="rearaxle_rectangle")
    state=State(0,0,0,0)
    result=fixed_pose_intrusion(state,world,car,span_s=6,
                                 planning_padding_m=.12)
    assert result["minimum_fixed_gap_m"] < 0
    assert 0 < result["fixed_planning_envelope_intrusion_s"] < result["fixed_physical_intrusion_s"] < 6


def test_short_horizon_has_no_physical_intrusion_but_might_cross_envelope():
    world=World((MovingObstacle("cyclist",3,0,-.5,0,.2),))
    car=replace(VehicleConfig(), road_footprint="rearaxle_rectangle")
    result=fixed_pose_intrusion(State(0,0,0,0),world,car,span_s=2.0)
    assert result["fixed_physical_intrusion_s"] is None
    with pytest.raises(ValueError, match="stationary"):
        fixed_pose_intrusion(State(0,0,0,0),world,car,span_s=-.1)


def test_empty_obstacle_has_no_false_stationary_collision():
    result=fixed_pose_intrusion(State(0,0,0,0), World(()),VehicleConfig())
    assert result["fixed_physical_intrusion_s"] is None
    assert result["fixed_planning_envelope_intrusion_s"] is None


def test_prefix_comparison_excludes_terminal_command_but_checks_physical_state():
    old=[{"time":0,"x":0,"y":0,"yaw":0,"speed":1,"steer":0,"command_speed":2},
         {"time":.1,"x":.1,"y":0,"yaw":0,"speed":1,"steer":0,"command_speed":""}]
    new=[dict(old[0]),dict(old[1],command_speed=2),
         {"time":.2,"x":.2,"y":0,"yaw":0,"speed":1,"steer":0}]
    report=prefix_comparison(old,new,cutoff_s=.1)
    assert report["prefix_equal"] and report["shared_frames"]==2
    corrupt=[dict(r) for r in new]
    corrupt[1]["y"] = .01
    assert not prefix_comparison(old,corrupt,cutoff_s=.1)["prefix_equal"]


def test_traces_keep_road_and_obstacle_gaps_and_rejections():
    trial=make_trial(TrialSpec("two_targets",0,"weak_brake"))
    result=run("two_targets",0,"two_stage_bodyframe",
               vehicle=trial.vehicle,planner=trial.planner,
               world_override=trial.world,
               config=RunConfig(max_time=.25,
                    physical_road_footprint="rearaxle_rectangle",
                    physical_body_offset_m=.55,goal_criterion="pose_gate"))
    physical=replace(trial.vehicle,road_footprint="rearaxle_rectangle")
    original=list(result.trace)
    frames=annotate_trace(result.trace,trial.world,physical)
    assert result.trace==original
    assert "physical_obstacle_gap_m" in frames[0]
    assert "fixed_physical_intrusion_s" not in frames[0] # ego moving
    assert "reject_hold_obstacle" in frames[0]
    out=windows(frames)
    assert len(out)==1
    assert "fraction_forward_feasible" in out[0]


def test_short_replay_writes_full_csv_and_verifies_horizon_invariance(tmp_path):
    report=study(seeds=(0,),horizons=(.2,.3),output=tmp_path,
                 physical_offset_m=.55,assumed_offset_m=.55)
    assert len(report["episodes"])==2
    assert len(report["prefix_checks"])==1
    assert report["prefix_checks"][0]["prefix_equal"]
    assert (tmp_path/"horizon_episodes.csv").exists()
    assert (tmp_path/"horizon_windows.csv").exists()
    assert (tmp_path/"trajectories"/"seed0_horizon0.2.csv").exists()
    assert (tmp_path/"study.json").exists()
