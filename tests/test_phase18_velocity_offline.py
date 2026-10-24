"""Fixed-ego offline estimator audit is a different estimand from robot safety."""
import csv
import math

import pytest

from lidar_sim.evaluation import TrialSpec
from lidar_sim.phase18_velocity_offline import (
    evaluate, fixed_ego_trajectory, save_csv,
)


def test_fixed_ego_identical_detections_and_no_controller_dynamics(tmp_path):
    specs = [TrialSpec("two_targets", 50, "weak_brake")]
    rows, details = evaluate(specs, dt=.1, ticks=17)
    assert len(rows) == 2
    assert details
    assert {r["reference_sensor"] for r in rows} == {
        "noisy", "degraded"}
    for r in rows:
        assert r["observation_seed"] > 0
        assert r["ticks"] == 17
        assert r["blocked_ticks_both"] >= 1
        assert r["scored_track_ticks_both"] > 0
        assert math.isfinite(float(r["ema_mean_velocity_error_mps"]))
        assert math.isfinite(float(r["ls5_mean_velocity_error_mps"]))
    assert all(row["identical_noisy_centers"] == 1 for row in details)
    assert all(row["fixed_ego_not_closed_loop"] == 1 for row in details)
    assert {row["horizon_s"] for row in details} == {.5, 1., 2.}
    output = tmp_path / "offline.csv"
    save_csv(output, rows)
    with output.open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_fixed_ego_deterministic_for_seed_50():
    case = [TrialSpec("two_targets", 50, "weak_brake")]
    assert evaluate(case, dt=.1, ticks=20) == evaluate(case, dt=.1, ticks=20)


def test_noisy_filter_has_identical_observation_counts_when_scoring():
    rows, details = evaluate(
        [TrialSpec("crossing", 51, "nominal")],
        dt=.1, ticks=50)
    assert len(rows) == 2
    assert all(r["scored_track_ticks_both"] >= 20 for r in rows)
    assert len(details) == sum(
        r["scored_track_ticks_both"]*3 for r in rows)


@pytest.mark.parametrize("dt,ticks", [
    (0, 2), (float("nan"), 2), (.1, 1),
    (.1, 0), (.1, True), (.1, 2.5),
])
def test_fixed_trajectory_invalid_requests_fail(dt, ticks):
    with pytest.raises(ValueError):
        fixed_ego_trajectory(dt=dt, ticks=ticks)


def test_missing_episodes_rejected():
    with pytest.raises(ValueError, match="at least one case"):
        evaluate([], dt=.1, ticks=10)
