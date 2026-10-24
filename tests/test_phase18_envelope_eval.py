"""Safety-envelope simulation must never use truth inside planner actions."""
from dataclasses import replace
import csv
import math

import pytest

from lidar_sim.evaluation import TrialSpec
from lidar_sim.experiment import RunConfig, run
from lidar_sim.model import State
from lidar_sim.phase18_envelope_eval import (
    calibrate, pair_rows, rank_quantile, run_study, POLICIES,
)
from lidar_sim.simulated_perception import PerceptionConfig, SimulatedPerception
from lidar_sim.world import MovingObstacle, World


def straight(n):
    return [State(0, 0, 0, 0, time=0.1*i) for i in range(n)]


def test_inflation_only_changes_estimated_radius_and_not_true_geometry():
    truth = World((MovingObstacle("person", 12, -1, 0, .3, .35),))
    cfg = PerceptionConfig(position_sigma_m=0, range_sigma_per_m=0,
                           dropout_probability=0, capture_delay_ticks=0)
    a = SimulatedPerception(cfg, 14, obstacle_padding_m=0)
    b = SimulatedPerception(cfg, 14, obstacle_padding_m=.6)
    for n in (1, 2, 3):
        first = a.observe(truth, straight(n), dt=.1)
        second = b.observe(truth, straight(n), dt=.1)
        assert first.blocked_reason == second.blocked_reason
        if first.valid_tracks:
            old = first.estimated_world.obstacles[0]
            padded = second.estimated_world.obstacles[0]
            assert padded.radius == pytest.approx(old.radius+.6)
            assert (old.x0, old.y0, old.vx, old.vy,
                    old.reference_time_s) == (
                    padded.x0, padded.y0, padded.vx, padded.vy,
                    padded.reference_time_s)
    assert truth.obstacles[0].radius == .35


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), True])
def test_invalid_padding_fails_closed(invalid):
    with pytest.raises(ValueError, match="obstacle_padding"):
        SimulatedPerception(PerceptionConfig(), 1, obstacle_padding_m=invalid)


def test_episode_quantile_rank_requires_enough_independent_cases():
    assert rank_quantile([.1]*9, .1) == pytest.approx(.1)
    assert rank_quantile(list(range(10)), .1) == 9
    with pytest.raises(ValueError, match="insufficient"):
        rank_quantile(list(range(8)), .1)
    with pytest.raises(ValueError, match="nonempty"):
        rank_quantile([.2, math.nan], .1)
    with pytest.raises(ValueError, match="alpha"):
        rank_quantile([1.], 0)


def test_calibration_only_uses_predeclared_simulator_seed_split():
    cal = [TrialSpec("two_targets", seed, "weak_brake")
           for seed in range(10, 19)]
    result = calibrate(cal, ticks=18, dt=.1)
    assert len(result["calibration_case_ids"]) == 9
    assert result["scored_horizons_s"] == [0.5, 1.0, 2.0]
    assert set(result["profiles"]) == {"noisy_ls5", "degraded_ls5"}
    for name, p in result["profiles"].items():
        assert p["offline_source_profile"] == name.removesuffix("_ls5")
        assert p["offline_estimator_field"] == "ls5_cv_position_error_m"
        assert p["calibration_episode_units"] == 9
        assert p["scored_horizon_observations"] > 0
        assert math.isfinite(p["extra_radius_m"])
        assert p["extra_radius_m"] >= 0
    with pytest.raises(ValueError, match="unique"):
        calibrate(cal+cal[:1], ticks=18, dt=.1)


def test_evaluation_rejects_calibration_leakage_even_in_small_smoke(tmp_path):
    cal_ids = ["two_targets__weak_brake__60"]
    calibr = {"calibration_case_ids": cal_ids,
              "profiles": {
                  "noisy_ls5": {"extra_radius_m": .75},
                  "degraded_ls5": {"extra_radius_m": 1.0}}}
    with pytest.raises(ValueError, match="leakage"):
        run_study([TrialSpec("two_targets", 60, "weak_brake")],
                  calibration=calibr, output=tmp_path)
    rows = run_study([TrialSpec("two_targets", 61, "weak_brake")],
                     calibration=calibr, output=tmp_path,
                     profiles=("noisy_ls5",), max_time=.2)
    assert len(rows) == 3
    assert set(row["policy"] for row in rows) == set(POLICIES)
    assert [row["extra_obstacle_radius_m"] for row in rows] == [
        0., .3, .75]
    assert all(row["perception_enabled"] for row in rows)
    assert len(set(row["observation_seed"] for row in rows)) == 1
    pairs = pair_rows(rows)
    assert len(pairs) == 3
    assert len({p["baseline_collision"] for p in pairs}) == 1
    assert all(p["mode"] == "two_stage_viability" for p in pairs)
    with (tmp_path / "episodes.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 3


def test_physical_collision_cannot_be_hidden_by_envelope_inflation():
    truth = World((MovingObstacle("crash", 0, 0, 0, 0, .4),))
    s = SimulatedPerception(PerceptionConfig(dropout_probability=1),
                            9, obstacle_padding_m=10.)
    actual = run("empty", mode="two_stage_viability",
                 config=RunConfig(max_time=.2), world_override=truth,
                 perception=s)
    assert actual.collision
    assert not actual.success
    assert truth.obstacles[0].radius == .4


def test_pair_analysis_fails_on_incomplete_policy_triple():
    with pytest.raises(ValueError, match="incomplete"):
        pair_rows([{
            "case_id": "a", "mode": "two_stage_viability",
            "sensor_profile": "noisy_ls5", "policy": "none",
            "observation_seed": 1}])


def test_calibration_radius_order_does_not_need_to_beat_fixed_padding():
    # It is legitimate for the learned diagnostic radius to be tiny or huge.
    assert rank_quantile([.05]*10, .1) < .3
    assert rank_quantile([4.]*10, .1) > .3
