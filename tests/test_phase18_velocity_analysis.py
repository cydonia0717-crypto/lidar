"""End-to-end integrity checks for paired synthetic velocity-filter results."""
import csv

import pytest

from lidar_sim.evaluation import TrialSpec
from lidar_sim.phase18_sim_eval import PROFILES, run_study
from lidar_sim.phase18_velocity_analysis import evaluate


def test_matched_lag_and_noise_profiles_only_change_filter():
    for old, new in (("noisy", "noisy_ls5"),
                     ("degraded", "degraded_ls5")):
        left = PROFILES[old]
        right = PROFILES[new]
        assert right.velocity_estimator == "windowed_ls"
        assert right.regression_window == 5
        assert left.position_sigma_m == right.position_sigma_m
        assert left.range_sigma_per_m == right.range_sigma_per_m
        assert left.dropout_probability == right.dropout_probability
        assert left.capture_delay_ticks == right.capture_delay_ticks
        assert left.max_track_age_s == right.max_track_age_s


def test_paired_audit_requires_same_case_and_scored_trace_denominators(tmp_path):
    root = tmp_path / "ablation"
    result = run_study(
        [TrialSpec("crossing", 50, "nominal")],
        modes=("two_stage_viability",),
        profiles=("noisy", "noisy_ls5"),
        max_time=.4, output=root)
    assert len(result) == 2
    rows = evaluate(root)
    assert len(rows) == 1
    row = rows[0]
    assert row["seed"] == "50"
    assert row["reference_profile"] == "noisy"
    assert row["candidate_profile"] == "noisy_ls5"
    assert row["observation_seed"]
    assert row["reference_perception_blocked_steps"] >= 1
    assert row["candidate_perception_blocked_steps"] >= 1
    assert row["reference_decision_ticks"] <= 4
    assert row["candidate_decision_ticks"] <= 4
    for variant in ("reference", "candidate"):
        assert (row[variant+"_perception_eval_mean_velocity_error_mps_scored_ticks"]
                <= row[variant+"_decision_ticks"])


def test_missing_filter_pair_is_not_silently_accepted(tmp_path):
    root = tmp_path / "one"
    run_study([TrialSpec("crossing", 60, "nominal")],
              modes=("two_stage_bodyframe",),
              profiles=("noisy",), max_time=.1, output=root)
    with pytest.raises(ValueError, match="missing paired estimator"):
        evaluate(root)


def test_pair_rejects_non_matching_observation_seed(tmp_path):
    root = tmp_path / "broken"
    run_study([TrialSpec("crossing", 61, "nominal")],
              modes=("two_stage_bodyframe",),
              profiles=("noisy", "noisy_ls5"),
              max_time=.1, output=root)
    path = root / "episodes.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[1]["observation_seed"] = "123"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="different synthetic observation seeds"):
        evaluate(root)
