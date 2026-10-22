"""Fast viability must preserve EVERY hard check and the exhaustive decision."""
from dataclasses import replace
from pathlib import Path
import csv

import pytest

from lidar_sim.model import Command, State, VehicleConfig
from lidar_sim.world import World, MovingObstacle
from lidar_sim.planner import PlannerConfig, choose_action
from lidar_sim.safety import CommandDelayLine
from lidar_sim.two_stage import choose_two_stage,exists_strict_continuation
from lidar_sim.viability import probe_future
from lidar_sim.equivalence_audit import audit_pair, audit_folder


def line(state:State,cfg:PlannerConfig):
    return CommandDelayLine(Command(state.speed,state.steer),cfg.control_delay_s)


@pytest.mark.parametrize("world,state,braking",[
    (World(()),State(0,0,0,1.5),1.6),
    (World((MovingObstacle("cross",7.5,-1.5,0,.8,.45),)),State(1,0,0,2),1.6),
    (World((MovingObstacle("overlap",0,0,0,0,1),)),State(0,0,0,1.5),.9),
    (World((),road_half_width=1.2),State(0,.8,.6,1),.9),
    (World((MovingObstacle("ped",11,1,0,-.4,.4),)),State(3,.4,.1,1.5),.9),
])
def test_existence_matches_original_entire_strict_search(world,state,braking):
    cfg=PlannerConfig()
    car=replace(VehicleConfig(braking=braking),road_footprint="rearaxle_rectangle")
    history=line(state,cfg)
    exhaustive=choose_two_stage(state,world,car,cfg,history)
    yes=exists_strict_continuation(state,world,car,cfg,history)
    assert yes == bool(exhaustive.feasible_count and not exhaustive.unsafe_fallback)
    assert history.pending==[]


@pytest.mark.parametrize("world",[
    World(()),
    World((MovingObstacle("cyclist",9,-3.0,0,.6,.45),)),
    World((MovingObstacle("unsafe",0,0,0,0,.9),)),
    World((),road_half_width=1.2),
])
def test_old_and_fast_first_command_and_checked_trajectory_equal(world):
    cfg=PlannerConfig()
    car=VehicleConfig(braking=.9)
    s=State(1,0,0,1.5)
    l=line(s,cfg)
    slow=choose_action(s,world,car,cfg,mode="two_stage_viability",delay_line=l)
    fast=choose_action(s,world,car,cfg,mode="two_stage_viability_fast",delay_line=l)
    assert fast.command==slow.command
    assert fast.trajectory==slow.trajectory
    assert fast.unsafe_fallback==slow.unsafe_fallback
    assert fast.feasible_count==slow.feasible_count
    assert fast.rejection_counts==slow.rejection_counts
    assert fast.score==slow.score
    assert fast.cost_parts.get("lookahead_change")==slow.cost_parts.get("lookahead_change")
    assert fast.cost_parts["lookahead_existence_only"]==1
    assert slow.cost_parts["lookahead_existence_only"]==0


def test_fast_probe_does_not_change_actuator_line():
    state=State(0,0,0,1.5)
    cfg=PlannerConfig()
    car=VehicleConfig(braking=.9)
    original=line(state,cfg)
    original.enqueue(-.1,Command(.6,.2))
    before=original.copy()
    path,predicted=probe_future(state,Command(2.4,-.3),car,cfg,original)
    assert original.pending==before.pending
    assert original.active==before.active
    a=choose_action(state,World(()),car,cfg,
                    mode="two_stage_viability_fast",delay_line=original)
    assert original.pending==before.pending
    assert original.active==before.active


def test_action_equivalence_csv_detects_any_state_or_control_difference(tmp_path):
    row={"time":"0.0","x":"0.0","y":"0.0","yaw":"0.0",
         "speed":"1.5","steer":"0.0","command_speed":"1.8",
         "command_steer":"0.21","applied_speed":"1.5",
         "applied_steer":"0","body_center_x":"0.55",
         "body_center_y":"0.0","pose_gate_met":"0"}
    assert audit_pair([row],[dict(row)])["equivalent"]
    assert not audit_pair([row],[dict(row,command_steer="0.42")])["equivalent"]
    assert not audit_pair([row],[dict(row,x="1")])["equivalent"]
    assert not audit_pair([row],[row,row])["equivalent"]
    folder=tmp_path/"runs"
    trace_folder=folder/"all_traces"
    trace_folder.mkdir(parents=True)
    for mode in ("two_stage_viability","two_stage_viability_fast"):
        with (trace_folder/f"empty__nominal__0__{mode}.csv").open(
                "w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    assert audit_folder(folder)[0]["equivalent"]
    (trace_folder/"empty__nominal__0__two_stage_viability_fast.csv").unlink()
    with pytest.raises(ValueError,match="missing mode"):
        audit_folder(folder)


def test_validation_has_same_errors_on_fast_and_exhaustive():
    state=State(0,0,0,1.5); car=VehicleConfig()
    for mode in ("two_stage_viability","two_stage_viability_fast"):
        with pytest.raises(ValueError,match="top_k"):
            choose_action(state,World(()),car,replace(PlannerConfig(),viability_top_k=0),mode=mode)
        with pytest.raises(ValueError,match="probe_s"):
            choose_action(state,World(()),car,replace(PlannerConfig(),viability_probe_s=.1),mode=mode)


def test_planner_latency_distribution_is_reported_with_real_100ms_budget():
    from lidar_sim.experiment import RunConfig, run
    for mode in ("two_stage_viability","two_stage_viability_fast"):
        result=run("empty",mode=mode,
                   config=RunConfig(max_time=.5,
                       physical_road_footprint="rearaxle_rectangle",
                       physical_body_offset_m=.55,goal_criterion="pose_gate"))
        timings=[r["planning_ms"] for r in result.trace if "planning_ms" in r]
        assert timings
        assert result.planning_period_budget_ms==pytest.approx(100)
        assert result.planning_max_ms==max(timings)
        assert 0 <= result.planning_over_period_fraction <= 1
        assert result.planning_p95_ms <= result.planning_p99_ms
        assert result.planning_p99_ms <= result.planning_max_ms
        assert sum(t>100 for t in timings)/len(timings) == pytest.approx(
            result.planning_over_period_fraction)
