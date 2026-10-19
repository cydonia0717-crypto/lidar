"""Guard that first-stage progress is executable and reward never bypasses safety."""
import csv
import json

import pytest

from lidar_sim.model import State, VehicleConfig
from lidar_sim.world import MovingObstacle, World
from lidar_sim.planner import PlannerConfig, choose_action
from lidar_sim.stall_diagnostics import stall_windows
from lidar_sim.evaluation import main


def test_zero_commitment_weight_identical_to_original_two_stage():
    cfg = PlannerConfig(first_stage_progress_weight=0.0)
    car = VehicleConfig()
    state = State(0, 0, 0, 1.5)
    world = World((MovingObstacle("ped", 13.0, -4.4, 0, .58),))
    original = choose_action(state, world, car, cfg, mode="two_stage")
    committed = choose_action(state, world, car, cfg, mode="two_stage_commit")
    assert original.command == committed.command
    assert original.score == pytest.approx(committed.score)
    assert original.feasible_count == committed.feasible_count


def test_commitment_component_matches_selected_actual_first_segment():
    cfg = PlannerConfig(first_stage_progress_weight=7.0)
    plan = choose_action(State(0, 0, 0, 1.5), World(()),
                         VehicleConfig(), cfg, mode="two_stage_commit")
    assert not plan.unsafe_fallback
    assert "cost_first_stage_commitment" in plan.cost_parts
    assert "planned_first_progress_m" in plan.cost_parts
    assert "planned_second_progress_m" in plan.cost_parts
    actual_first = plan.cost_parts["planned_first_progress_m"]
    assert actual_first >= 0
    assert plan.cost_parts["cost_first_stage_commitment"] == pytest.approx(
        -cfg.first_stage_progress_weight * actual_first)
    chosen = next(row for row in plan.diagnostics if row["selected"])
    assert chosen["total_cost"] == pytest.approx(plan.score)
    assert chosen["cost_first_stage_commitment"] == pytest.approx(
        plan.cost_parts["cost_first_stage_commitment"])


def test_commitment_keeps_strict_collision_and_road_guards():
    cfg = PlannerConfig(first_stage_progress_weight=5000.0)
    state = State(0, 0, 0, 1.5)
    car = VehicleConfig()
    impossible = World((MovingObstacle("obstacle", 0, 0, 0, 0, 1),))
    plan = choose_action(state, impossible, car, cfg, mode="two_stage_commit")
    assert plan.unsafe_fallback
    assert plan.emergency
    assert plan.fallback_strategy == "unsafe_committed_prefix"


def test_stall_diagnostic_separates_executed_and_promised_motion():
    trace = [
        {"time": .0, "x": 2, "speed": 0, "command_speed": 0,
         "emergency": 0, "planned_second_speed": 1.8,
         "planned_first_progress_m": 0, "planned_second_progress_m": 1.2},
        {"time": .1, "x": 2, "speed": 0, "command_speed": 0,
         "emergency": 0, "planned_second_speed": 1.8,
         "planned_first_progress_m": 0, "planned_second_progress_m": 1.2},
    ]
    windows = stall_windows(trace, window_s=5)
    assert len(windows) == 1
    assert windows[0]["zero_first_command_fraction"] == 1
    assert windows[0]["planned_second_mean_speed_mps"] == 1.8
    assert windows[0]["planned_second_progress_mean_m"] == 1.2
    assert windows[0]["x_displacement_m"] == 0


def test_historical_and_heldout_seed_sets_are_distinct(tmp_path):
    original = tmp_path / "old"
    holdout = tmp_path / "holdout"
    argv = ["--scenarios", "empty", "--stresses", "nominal",
            "--modes", "two_stage", "--seeds", "1", "--max-time", ".2"]
    main(argv + ["--seed-start", "0", "--output", str(original)])
    main(argv + ["--seed-start", "10", "--output", str(holdout)])
    def inspect(out):
        with (out / "episodes.csv").open(encoding="utf-8") as fd:
            return list(csv.DictReader(fd))
    assert inspect(original)[0]["case_id"] != inspect(holdout)[0]["case_id"]
    with (holdout / "metadata.json").open(encoding="utf-8") as fd:
        metadata = json.load(fd)
    assert metadata["design"]["seed_start"] == 10


def test_negative_first_stage_weight_rejected():
    with pytest.raises(ValueError, match="first-stage"):
        choose_action(State(0,0,0,0), World(()), VehicleConfig(),
                      PlannerConfig(first_stage_progress_weight=-1),
                      mode="two_stage_commit")


def test_online_trace_reports_feasible_moving_commands_separately():
    from lidar_sim.experiment import run, RunConfig
    result = run("empty", mode="two_stage", config=RunConfig(max_time=.3))
    first = result.trace[0]
    assert first["feasible_pairs"] > 0
    assert first["max_feasible_first_speed"] > 0
    assert first["moving_first_action_feasible"] == 1
    assert first["stopped_choice_with_moving_option"] == int(
        first["command_speed"] < .1 and first["moving_first_action_feasible"] == 1)
    rows = stall_windows(result.trace)
    assert rows and rows[0]["mean_feasible_pairs"] > 0


def test_stall_diagnosis_distinguishes_choice_vs_strict_infeasibility():
    rows = [
        {"time": 0, "x": 5., "speed": 0., "command_speed": 0.,
         "emergency": 0, "feasible_pairs": 4,
         "moving_first_action_feasible": 1,
         "stopped_choice_with_moving_option": 1},
        {"time": .1, "x": 5., "speed": 0., "command_speed": 0.,
         "emergency": 1, "feasible_pairs": 0,
         "moving_first_action_feasible": 0,
         "stopped_choice_with_moving_option": 0},
    ]
    window = stall_windows(rows)[0]
    assert window["moving_first_action_feasible_fraction"] == .5
    assert window["stop_despite_moving_first_action_fraction"] == .5
    assert window["mean_feasible_pairs"] == 2
    assert window["emergency_fraction"] == .5
