"""Two-stage search regression: strict constraints + planned continuation only."""
import csv

from lidar_sim.model import Command, State, VehicleConfig
from lidar_sim.safety import CommandDelayLine
from lidar_sim.two_stage import continuation_options
from lidar_sim.planner import PlannerConfig, choose_action, collision_free
from lidar_sim.experiment import RunConfig, run, save_result
from lidar_sim.world import World, MovingObstacle


def test_second_action_candidates_are_unique():
    car = VehicleConfig()
    first = Command(1.2, .42)
    options = continuation_options(first, car)
    assert len(options) == len(set(options))
    assert first in options
    assert Command(0, .42) in options
    assert Command(1.2, -.42) in options


def test_two_stage_empty_world_finds_braking_feasible_path():
    car = VehicleConfig()
    cfg = PlannerConfig()
    start = State(0, 0, 0, 1.5)
    plan = choose_action(start, World(()), car, cfg, mode="two_stage")
    assert not plan.emergency
    assert plan.backup_checked
    assert plan.feasible_count > 0
    assert plan.candidate_count >= plan.feasible_count
    assert collision_free(plan.trajectory, World(()), car,
                          padding=cfg.prediction_padding)
    assert plan.trajectory[-1].speed == 0
    assert plan.trajectory[-1].time > cfg.control_delay_s + cfg.candidate_execute_s + cfg.two_stage_second_s
    assert any(row["selected"] for row in plan.diagnostics)
    assert "cost_action_switch" in plan.cost_parts


def test_committed_prefix_collision_cannot_be_bypassed():
    car = VehicleConfig()
    cfg = PlannerConfig()
    first = State(0, 0, 0, 0.0)
    world = World((MovingObstacle("overlap", 0, 0, 0, 0, 1),))
    plan = choose_action(first, world, car, cfg, mode="two_stage")
    assert plan.emergency
    assert plan.unsafe_fallback
    assert plan.fallback_strategy == "unsafe_committed_prefix"


def test_two_stage_does_not_invent_undelayed_actuation():
    car = VehicleConfig()
    cfg = PlannerConfig(control_delay_s=.3)
    line = CommandDelayLine(Command(1, 0), .3)
    line.enqueue(-.1, Command(0, 0))
    state = State(0, 0, 0, 1)
    plan = choose_action(state, World(()), car, cfg,
                         mode="two_stage", delay_line=line)
    assert plan.trajectory[0] == state
    assert plan.trajectory[-1].time >= .3 + .4 + cfg.two_stage_second_s
    assert line.pending  # virtual prediction must not consume actual commands


def test_two_stage_decision_csv_handles_different_candidate_columns(tmp_path):
    car = VehicleConfig()
    initial = State(0, 0, 0, 1.5)
    cfg = PlannerConfig()
    world = World((MovingObstacle("ped", 13, -4.4, 0, .58),))
    p = choose_action(initial, world, car, cfg, mode="two_stage")
    assert len(p.diagnostics) == 25
    output = run("empty", mode="two_stage", config=RunConfig(max_time=.2))
    output.decision_snapshot = {"rows": p.diagnostics}
    save_result(output, tmp_path)
    with (tmp_path / "empty_two_stage_0_decision_candidates.csv").open(
            newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 25
    assert "second_speed" in rows[0]


def test_all_actions_physically_infeasible_are_reported():
    car = VehicleConfig()
    cfg = PlannerConfig()
    state = State(0, 0, 0, 1)
    world = World((), road_half_width=.5)
    plan = choose_action(state, world, car, cfg, mode="two_stage")
    assert plan.unsafe_fallback
    assert plan.emergency
    assert not plan.feasible_count
