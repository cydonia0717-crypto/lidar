"""Unbiased episode pairing, uncertainty intervals and output tests."""
from __future__ import annotations

import csv
import json

import pytest

from lidar_sim.evaluation import (
    TrialSpec, evaluate_cases, make_trial, paired_braking_effects, plot_study,
    save_study, summarize, wilson_95, parse_names, VALID_STRESSES)
from lidar_sim.experiment import RunConfig, run
from lidar_sim.model import VehicleConfig
from lidar_sim.world import MovingObstacle, World


def test_same_trial_is_deterministic_and_physics_changes_by_stress():
    nominal = make_trial(TrialSpec("crossing", 11, "nominal"))
    again = make_trial(TrialSpec("crossing", 11, "nominal"))
    weak = make_trial(TrialSpec("crossing", 11, "weak_brake"))
    fast = make_trial(TrialSpec("crossing", 11, "fast_targets"))
    delayed = make_trial(TrialSpec("crossing", 11, "high_delay"))
    assert nominal == again
    assert weak.world == nominal.world
    assert weak.vehicle.braking < nominal.vehicle.braking
    assert fast.world.obstacles[0].vy == pytest.approx(
        nominal.world.obstacles[0].vy * 1.5)
    assert delayed.planner.control_delay_s > nominal.planner.control_delay_s


def test_physical_world_override_is_used_in_run():
    normal = run("empty", seed=0, config=RunConfig(max_time=.2))
    blocked_world = World((MovingObstacle("blocking", 0, 0, 0, 0, 1),))
    blocked = run("empty", seed=0, config=RunConfig(max_time=.2),
                  world_override=blocked_world)
    assert not normal.collision
    assert blocked.collision


def test_paired_trial_has_identical_environment_parameters():
    records = evaluate_cases([TrialSpec("empty", 4, "high_delay")],
                             ("predictive", "delay_aware", "braking"),
                             max_time=.3)
    assert len(records) == 3
    assert len({r["case_id"] for r in records}) == 1
    for key in ("target_speed_scale", "road_half_width_m",
                "control_delay_s", "braking_mps2", "scenario", "seed"):
        assert len({r[key] for r in records}) == 1
    assert {r["mode"] for r in records} == {"predictive", "delay_aware", "braking"}


def test_wilson_intervals_have_coverage_bounds():
    lo, hi = wilson_95(0, 10)
    assert lo == 0
    assert 0 < hi < .5
    lo, hi = wilson_95(10, 10)
    assert .5 < lo < 1
    assert hi == 1
    lo, hi = wilson_95(5, 10)
    assert lo < .5 < hi
    with pytest.raises(ValueError):
        wilson_95(0, 0)


def test_success_time_does_not_average_collisions_and_timeouts():
    rows = [
        {"scenario": "crossing", "stress": "nominal", "mode": "braking",
         "success": True, "collision": False, "duration_s": 14.0,
         "min_clearance_m": .4, "stopped_steps": 2,
         "unsafe_fallback_steps": 0, "backup_rejections": 2,
         "mean_planning_ms": 5.0},
        {"scenario": "crossing", "stress": "nominal", "mode": "braking",
         "success": False, "collision": True, "duration_s": 2.0,
         "min_clearance_m": -.1, "stopped_steps": 0,
         "unsafe_fallback_steps": 1, "backup_rejections": 7,
         "mean_planning_ms": 5.0},
    ]
    summary = summarize(rows)[0]
    assert summary["n"] == 2
    assert summary["mean_success_time_s"] == 14.0
    assert summary["collision_rate"] == .5


def test_paired_differences_preserve_improvements_and_regressions():
    params = dict(case_id="crossing__nominal__0", scenario="crossing",
                  stress="nominal", seed=0, target_speed_scale=1.0,
                  road_half_width_m=3.25, control_delay_s=.25, braking_mps2=1.6)
    records = [
        dict(params, mode="delay_aware", success=True, collision=False,
             min_clearance_m=.4, stopped_steps=1, unsafe_fallback_steps=0,
             duration_s=12.0),
        dict(params, mode="braking", success=False, collision=True,
             min_clearance_m=-.1, stopped_steps=8, unsafe_fallback_steps=2,
             duration_s=6.0),
    ]
    effect = paired_braking_effects(records)[0]
    assert effect["collision_change"] == 1
    assert effect["success_change"] == -1
    assert effect["success_time_change_s"] == ""
    assert effect["braking_unsafe_fallback_steps"] == 2
    records[1]["road_half_width_m"] = 2.0
    with pytest.raises(ValueError, match="unpaired physics"):
        paired_braking_effects(records)


def test_reject_unknown_and_duplicate_cli_values():
    assert parse_names("nominal,weak_brake", VALID_STRESSES, "test") == (
        "nominal", "weak_brake")
    with pytest.raises(ValueError):
        parse_names("nominal,nominal", VALID_STRESSES, "test")
    with pytest.raises(ValueError):
        parse_names("not-a-stress", VALID_STRESSES, "test")


def test_save_audit_files_and_plot(tmp_path):
    cases = [TrialSpec("empty", 1, "nominal")]
    records = evaluate_cases(cases, ("delay_aware", "braking"), max_time=.3)
    files = save_study(records, tmp_path, inputs={"purpose": "unit-test"}, plot=True)
    assert files["episodes"].is_file()
    assert files["summary"].is_file()
    assert files["paired"].is_file()
    assert files["figure"].stat().st_size > 5000
    metadata = json.loads(files["metadata"].read_text(encoding="utf-8"))
    assert metadata["n_episodes"] == 2
    assert metadata["n_distinct_trials"] == 1
    assert "ideal CV" in metadata["source"]
    with files["paired"].open(newline="", encoding="utf-8") as fh:
        pairs = list(csv.DictReader(fh))
    assert len(pairs) == 1


def test_cli_smoke_creates_outputs(tmp_path):
    from lidar_sim.evaluation import main
    main(["--scenarios", "empty", "--stresses", "nominal",
          "--modes", "delay_aware,braking", "--seeds", "1",
          "--max-time", ".2", "--output", str(tmp_path)])
    assert (tmp_path / "episodes.csv").exists()
    assert (tmp_path / "summary_by_scenario_stress_mode.csv").exists()


def test_failed_episode_trace_is_saved_for_debugging(tmp_path):
    records = evaluate_cases(
        [TrialSpec("empty", 0, "nominal"),
         TrialSpec("empty", 1, "nominal")],
        ("delay_aware", "braking"), max_time=.2,
        failure_trace_dir=tmp_path,
        failure_traces_per_group=1,
    )
    # Both controllers time out because 0.2s is far too short to reach goal.
    # Per-group quota prevents thousands of debug traces in full studies.
    saved = [r for r in records if r["failure_trace_file"]]
    assert len(saved) == 2
    assert all((tmp_path / (r["case_id"] + "__" + r["mode"] + ".csv")).exists()
               for r in saved)
    assert all(not r["success"] for r in records)
    with (tmp_path / "empty__nominal__0__braking.csv").open(
            newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) >= 2


def test_csv_writer_handles_optional_diagnostic_columns(tmp_path):
    from lidar_sim.evaluation import _write_csv
    file = tmp_path / "mixed.csv"
    _write_csv(file, [{"time": 0.0, "emergency": 1},
                      {"time": .1, "cost_route": 2.0}])
    with file.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[1]["cost_route"] == "2.0"
    assert rows[0]["cost_route"] == ""
