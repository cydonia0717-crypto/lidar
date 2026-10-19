"""Conservative one-replanning-point viability counterfactual tests."""
from dataclasses import replace
from math import isfinite

import pytest

from lidar_sim.model import State, VehicleConfig, Command
from lidar_sim.world import World, MovingObstacle
from lidar_sim.planner import PlannerConfig, choose_action, collision_free
from lidar_sim.safety import CommandDelayLine
from lidar_sim.viability import probe_future
from lidar_sim.experiment import RunConfig, run


def configured():
    car=VehicleConfig()
    cfg=PlannerConfig()
    state=State(0,0,0,1.5)
    return state, car, cfg


def test_probe_respects_real_delay_and_does_not_consume_actual_queue():
    state,car,cfg=configured()
    original=CommandDelayLine(Command(1.5,0),cfg.control_delay_s)
    original.enqueue(-.05,Command(.6,.1))
    before=original.copy()
    candidate=Command(2.4,-.42)
    path,line=probe_future(state,candidate,car,cfg,original)
    assert original.active==before.active
    assert original.pending==before.pending
    assert path[0]==state
    assert path[-1].time==pytest.approx(cfg.viability_probe_s)
    assert original.pending[0].activate_at==pytest.approx(.20)
    assert path[-1].steer < 0.1 # candidate eventually enters delayed queue
    assert line.pending == [] # 0.60s reaches all activation boundaries


def test_original_two_stage_unchanged_when_other_mode_is_probed():
    state,car,cfg=configured()
    world=World(())
    baseline=choose_action(state,world,car,cfg,mode="two_stage_bodyframe")
    proposed=choose_action(state,world,car,cfg,mode="two_stage_viability")
    repeated=choose_action(state,world,car,cfg,mode="two_stage_bodyframe")
    assert baseline.command==repeated.command
    assert baseline.score==repeated.score
    assert proposed.backup_checked and not proposed.unsafe_fallback
    assert proposed.cost_parts["lookahead_probed"]>=1
    assert proposed.cost_parts["lookahead_viable"]>=1
    assert proposed.cost_parts["lookahead_enabled"]==1
    assert proposed.cost_parts["lookahead_no_viable_shortlist"]==0
    assert collision_free(proposed.trajectory,world,
                          replace(car,road_footprint="rearaxle_rectangle"),
                          padding=cfg.prediction_padding)


def test_requested_action_is_always_from_original_strictly_eligible_set():
    state,car,cfg=configured()
    world=World((MovingObstacle("crossing",13,-4.4,0,.58),))
    base=choose_action(state,world,car,cfg,mode="two_stage_bodyframe")
    probe=choose_action(state,world,car,cfg,mode="two_stage_viability")
    eligible={(r["target_speed"],r["target_steer"])
              for r in base.diagnostics
              if r["status"] in ("selected","eligible")}
    assert (probe.command.target_speed,probe.command.target_steer) in eligible
    assert probe.feasible_count==base.feasible_count
    assert probe.candidate_count==base.candidate_count
    assert probe.rejection_counts==base.rejection_counts
    assert sum(bool(r["selected"]) for r in probe.diagnostics)==1
    assert probe.cost_parts["lookahead_probed"]<=cfg.viability_top_k


def test_overlapping_committed_prefix_is_never_allowed_to_escape_safety():
    state,car,cfg=configured()
    world=World((MovingObstacle("overlap",0,0,0,0,1),))
    base=choose_action(state,world,car,cfg,mode="two_stage_bodyframe")
    probe=choose_action(state,world,car,cfg,mode="two_stage_viability")
    assert base.unsafe_fallback and probe.unsafe_fallback
    assert probe.fallback_strategy=="unsafe_committed_prefix"
    assert probe.command==base.command
    assert probe.cost_parts["lookahead_skipped_unsafe_baseline"]==1


def test_invalid_topk_or_window_rejected():
    state,car,cfg=configured()
    for k in (0,-1,2.3,True):
        with pytest.raises(ValueError,match="top_k"):
            choose_action(state,World(()),car,replace(cfg,viability_top_k=k),
                          mode="two_stage_viability")
    for value in (0,.2,.25,float("inf")):
        with pytest.raises(ValueError,match="probe_s"):
            choose_action(state,World(()),car,
                          replace(cfg,viability_probe_s=value),
                          mode="two_stage_viability")


def test_short_real_plant_uses_same_physical_truth_and_goal():
    config=RunConfig(max_time=.2,goal_criterion="pose_gate",
                     physical_road_footprint="rearaxle_rectangle",
                     physical_body_offset_m=.55)
    result=run("empty",mode="two_stage_viability",config=config)
    assert result.goal_criterion=="pose_gate"
    assert result.assumed_body_offset_m==pytest.approx(.55)
    assert result.physical_body_offset_m==pytest.approx(.55)
    assert result.trace[0]["lookahead_enabled"]==1
    assert result.trace[0]["lookahead_probed"]>=1
    assert not result.collision and not result.road_exit


def test_paired_viability_csv_and_all_episode_traces(tmp_path):
    import csv
    from pathlib import Path
    from lidar_sim.evaluation import (TrialSpec, evaluate_cases,
                                       paired_viability_effects, save_study)
    output=tmp_path/"study"
    rows=evaluate_cases(
        [TrialSpec("empty",0,"nominal")],
        ("two_stage_bodyframe","two_stage_viability"),
        max_time=.2,physical_road_footprint="rearaxle_rectangle",
        goal_criterion="pose_gate",
        all_trace_dir=output/"all_traces")
    pairs=paired_viability_effects(rows)
    assert len(pairs)==1
    assert pairs[0]["viability_mean_planning_ms"]>0
    assert pairs[0]["base_mean_planning_ms"]>0
    assert len(list((output/"all_traces").glob("*.csv")))==2
    for row in rows:
        assert Path(row["full_trace_file"]).exists()
    files=save_study(rows,output,
        inputs={"goal_criterion":"pose_gate"})
    with files["paired_viability"].open(encoding="utf8",newline="") as fh:
        assert len(list(csv.DictReader(fh)))==1
    mismatched=[rows[0],dict(rows[1],goal_lateral_tolerance_m=1.5)]
    with pytest.raises(ValueError,match="physical/goal"):
        paired_viability_effects(mismatched)
    with pytest.raises(ValueError,match="requires common"):
        paired_viability_effects([
            dict(r,goal_criterion="x_gate") for r in rows])
