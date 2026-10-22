"""Fast existence-only viability must preserve phase-15 actions exactly.

The new mode is an optimization of the *same* finite-depth heuristic, not a
change in dynamics, model dimensions, braking, delay, or any safety bound.
"""
from dataclasses import replace
import math
import random

import pytest

from lidar_sim.model import Command, State, VehicleConfig
from lidar_sim.world import World, MovingObstacle
from lidar_sim.planner import PlannerConfig, choose_action
from lidar_sim.safety import CommandDelayLine
from lidar_sim.two_stage import (choose_two_stage,
                                 has_strict_two_stage_successor)
from lidar_sim.evaluation import TrialSpec, make_trial
from lidar_sim.experiment import RunConfig, run


@pytest.mark.parametrize("seed", [0, 1, 10, 11, 40])
def test_exact_hard_filter_existence_matches_full_planning_on_fixed_synthetic_world(seed):
    trial=make_trial(TrialSpec("two_targets",seed,"weak_brake"))
    vehicle=replace(trial.vehicle,road_footprint="rearaxle_rectangle")
    line=CommandDelayLine(Command(1.2,0),trial.planner.control_delay_s)
    line.enqueue(-.05,Command(1.8, .21))
    states=(State(0,0,0,1.5),State(9.5,0,.08,1.2,time=4),
            State(11.2,.25,.23,.3,time=9),
            State(12.0,.7,-.14,0,time=13))
    for initial in states:
        full=choose_two_stage(initial,trial.world,vehicle,trial.planner,line)
        exists=has_strict_two_stage_successor(
            initial,trial.world,vehicle,trial.planner,line)
        assert exists==(full.feasible_count>0 and not full.unsafe_fallback)
    assert len(line.pending)==1  # both searches clone, not consume actuator


@pytest.mark.parametrize("scenario,stress,seed", [
    ("empty","nominal",0),
    ("two_targets","weak_brake",0),
    ("two_targets","weak_brake",1),
    ("two_targets","weak_brake",10),
    ("crossing","nominal",40),
    ("narrow","nominal",40),
])
def test_identical_selected_action_and_path_at_same_state(scenario,stress,seed):
    trial=make_trial(TrialSpec(scenario,seed,stress))
    state=State(6,0,.05,1.4,time=2.5)
    line=CommandDelayLine(Command(1.4,0),trial.planner.control_delay_s)
    baseline=choose_action(state,trial.world,trial.vehicle,trial.planner,
                           mode="two_stage_viability",delay_line=line)
    fast=choose_action(state,trial.world,trial.vehicle,trial.planner,
                       mode="two_stage_viability_fast",delay_line=line)
    assert fast.command==baseline.command
    assert fast.trajectory==baseline.trajectory
    assert fast.score==baseline.score
    assert fast.feasible_count==baseline.feasible_count
    assert fast.rejection_counts==baseline.rejection_counts
    assert fast.unsafe_fallback==baseline.unsafe_fallback
    assert fast.cost_parts["lookahead_probed"] <= baseline.cost_parts["lookahead_probed"]
    assert fast.cost_parts["lookahead_existential_fast"]==1
    assert len(line.pending)==0


@pytest.mark.parametrize("seed", [0,1])
def test_clipped_closed_loop_actions_are_bitwise_identical(seed):
    trial=make_trial(TrialSpec("two_targets",seed,"weak_brake"))
    settings=RunConfig(max_time=1.1,physical_road_footprint="rearaxle_rectangle",
                       goal_criterion="pose_gate",physical_body_offset_m=.55)
    slow=run("two_targets",seed,"two_stage_viability",vehicle=trial.vehicle,
             planner=trial.planner,world_override=trial.world,config=settings)
    fast=run("two_targets",seed,"two_stage_viability_fast",vehicle=trial.vehicle,
             planner=trial.planner,world_override=trial.world,config=settings)
    assert len(slow.trace)==len(fast.trace)
    for old,new in zip(slow.trace,fast.trace):
        for key in ("time","x","y","yaw","speed","steer",
                    "command_speed","command_steer",
                    "applied_speed","applied_steer"):
            assert old[key]==new[key]
    assert fast.collision==slow.collision
    assert fast.timeout==slow.timeout


def test_viability_fast_rejects_same_unsafe_committed_prefix():
    config=PlannerConfig()
    state=State(0,0,0,1.2)
    world=World((MovingObstacle("overlap",0,0,0,0,1.),))
    old=choose_action(state,world,VehicleConfig(),config,
                      mode="two_stage_viability")
    new=choose_action(state,world,VehicleConfig(),config,
                      mode="two_stage_viability_fast")
    assert old.unsafe_fallback and new.unsafe_fallback
    assert old.fallback_strategy==new.fallback_strategy
    assert old.command==new.command


def test_optimization_does_not_undo_existing_parameter_validation():
    cfg=PlannerConfig()
    car=VehicleConfig()
    state=State(0,0,0,1.5)
    for invalid in (0,-.1):
        with pytest.raises(ValueError,match="second-stage"):
            choose_action(state,World(()),car,replace(cfg,two_stage_second_s=invalid),
                          mode="two_stage_viability_fast")
    for invalid in (0,.20):
        with pytest.raises(ValueError,match="probe_s"):
            choose_action(state,World(()),car,replace(cfg,viability_probe_s=invalid),
                          mode="two_stage_viability_fast")
