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


def test_phase16_fast_exit_preserves_first_viable_action_and_records_skips(monkeypatch):
    """On an obstacle-free scene the first ranked candidate has a continuation.

    Once found, probing lower-ranked eligible candidates cannot alter the
    chosen first command. Metrics must distinguish skipped from rejected.
    """
    import lidar_sim.viability as viability

    state, car, cfg = configured()
    world = World(())
    line = CommandDelayLine(Command(state.speed, state.steer), cfg.control_delay_s)
    baseline = viability.choose_two_stage(state, world, car, cfg, line)
    eligible = sorted(
        (r for r in baseline.diagnostics
         if r["status"] in ("eligible", "selected")
         and r.get("continuation_count", 0) > 0
         and isfinite(r["total_cost"])),
        key=lambda r: (r["total_cost"], r["target_speed"], r["target_steer"])
    )[:cfg.viability_top_k]
    assert eligible

    calls = []
    actual = viability.choose_two_stage

    def counted(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)

    monkeypatch.setattr(viability, "choose_two_stage", counted)
    optimized = viability.choose_viability_probed(state, world, car, cfg, line)

    assert optimized.cost_parts["lookahead_probed"] == 1
    assert optimized.cost_parts["lookahead_viable"] == 1
    assert optimized.cost_parts["lookahead_unprobed"] == len(eligible) - 1
    assert optimized.cost_parts["lookahead_early_exit"] == int(len(eligible) > 1)
    assert optimized.command == Command(
        eligible[0]["target_speed"], eligible[0]["target_steer"])
    assert optimized.feasible_count == baseline.feasible_count
    assert len(calls) == 1  # only parent is fully scored; child uses witness
    assert optimized.cost_parts["lookahead_child_calls"] == 1
    assert optimized.cost_parts["lookahead_child_witness"] == 1
    assert optimized.cost_parts["lookahead_child_count_exact"] == 0


@pytest.mark.parametrize("obstacles", [
    (),
    (MovingObstacle("crossing", 13, -4.4, 0, .58),),
])
def test_phase16_cached_trajectory_matches_forced_first(obstacles):
    """Cache and original forced search agree on score, full path and costs."""
    from lidar_sim.two_stage import choose_two_stage

    state, car, cfg = configured()
    world = World(obstacles)
    line = CommandDelayLine(Command(state.speed, state.steer), cfg.control_delay_s)
    cache = {}
    baseline = choose_two_stage(
        state, world, car, cfg, line, feasible_first_cache=cache)
    assert not baseline.unsafe_fallback
    assert cache
    for command, (score, trajectory, parts) in cache.items():
        forced = choose_two_stage(
            state, world, car, cfg, line, forced_first=command)
        assert not forced.unsafe_fallback
        assert forced.command == command
        assert forced.score == score
        assert forced.trajectory == trajectory
        for key, value in parts.items():
            assert forced.cost_parts[key] == value
    assert baseline.command in cache


@pytest.mark.parametrize("state,world", [
    (State(0, 0, 0, 1.5), World(())),
    (State(0, 0, 0, 1.5),
     World((MovingObstacle("crossing", 13, -4.4, 0, .58),))),
    (State(0, 0, 0, 1.5),
     World((MovingObstacle("overlap", 0, 0, 0, 0, 1),))),
    (State(2, .2, .1, 1.5),
     World((MovingObstacle("near", 5, .5, -.2, 0, .6),))),
])
def test_strict_child_witness_agrees_with_exhaustive_two_stage(state, world):
    """Existence is true iff the original exhaustive planner finds a pair."""
    from lidar_sim.two_stage import choose_two_stage
    from lidar_sim.viability_witness import strict_continuation_exists

    car = VehicleConfig(road_footprint="rearaxle_rectangle")
    cfg = PlannerConfig()
    line = CommandDelayLine(Command(state.speed, state.steer), cfg.control_delay_s)
    line.enqueue(state.time - .1, Command(.8, .1))
    before = line.copy()
    full = choose_two_stage(state, world, car, cfg, line)
    witness = strict_continuation_exists(state, world, car, cfg, line)
    assert witness == bool(full.feasible_count and not full.unsafe_fallback)
    assert line.active == before.active
    assert line.pending == before.pending
