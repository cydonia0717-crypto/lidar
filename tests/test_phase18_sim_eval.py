"""Matched phase18 sensing ablation metadata and physical truth isolation."""
import csv
import json

import pytest

from lidar_sim.evaluation import TrialSpec
from lidar_sim.phase18_sim_eval import PROFILES, run_study


def test_short_closed_loop_study_writes_all_profiles_and_traces(tmp_path):
    root = tmp_path / "run"
    records = run_study(
        [TrialSpec("crossing", 0, "nominal")],
        modes=("two_stage_viability",),
        profiles=("oracle", "noisy"),
        max_time=.2, output=root)
    assert len(records) == 2
    assert all(row["case_id"] == "crossing__nominal__0" for row in records)
    by_profile = {r["perception_profile"]: r for r in records}
    assert not by_profile["oracle"]["perception_enabled"]
    assert by_profile["noisy"]["perception_enabled"]
    assert by_profile["noisy"]["perception_blocked_steps"] >= 1
    assert by_profile["noisy"]["perception_planned_steps"] <= 2
    assert by_profile["oracle"]["observation_seed"] == by_profile["noisy"]["observation_seed"]
    assert by_profile["oracle"]["physical_body_offset_m"] == (
        by_profile["noisy"]["physical_body_offset_m"])
    with (root / "episodes.csv").open(newline="", encoding="utf-8") as f:
        recorded = list(csv.DictReader(f))
    assert len(recorded) == 2
    for p in ("oracle", "noisy"):
        directory = root / "crossing__nominal__0" / p / "two_stage_viability"
        assert len(list(directory.glob("*_trace.csv"))) == 1
        assert len(list(directory.glob("*_metrics.json"))) == 1
    metadata = json.loads((root / "inputs.json").read_text())
    assert metadata["profiles"]["oracle"] == "exact oracle target truth"
    assert metadata["profiles"]["noisy"]["capture_delay_ticks"] == 1
    assert any("no PointCloud2" in issue for issue in metadata["model_limitations"])


def test_equal_sensing_seeds_across_mode_order(tmp_path):
    spec = [TrialSpec("crossing", 0, "nominal")]
    baseline = run_study(
        spec, modes=("two_stage_viability", "two_stage_bodyframe"),
        profiles=("noisy",), max_time=.2, output=tmp_path / "first")
    reversed_order = run_study(
        spec, modes=("two_stage_bodyframe", "two_stage_viability"),
        profiles=("noisy",), max_time=.2, output=tmp_path / "second")
    assert len(set(r["observation_seed"] for r in baseline)) == 1
    assert baseline[0]["observation_seed"] == reversed_order[0]["observation_seed"]
    a = {r["mode"]: (r["success"], r["collision"], r["duration_s"],
                      r["perception_blocked_steps"],
                      r["perception_detection_events"])
         for r in baseline}
    b = {r["mode"]: (r["success"], r["collision"], r["duration_s"],
                      r["perception_blocked_steps"],
                      r["perception_detection_events"])
         for r in reversed_order}
    assert a == b


@pytest.mark.parametrize("modes,profiles", [
    ((), ("noisy",)), (("bad",), ("noisy",)),
    (("two_stage_viability",), ("unreal",)),
    (("two_stage_viability",), ("noisy", "noisy")),
])
def test_invalid_experiment_dimensions_rejected(tmp_path, modes, profiles):
    with pytest.raises(ValueError):
        run_study([TrialSpec("crossing", 0, "nominal")],
                  modes=modes, profiles=profiles, max_time=.2,
                  output=tmp_path)


def test_profiles_have_explicit_noise_and_lag_configuration():
    assert PROFILES["oracle"] is None
    assert PROFILES["clean"].position_sigma_m == 0
    assert PROFILES["clean"].dropout_probability == 0
    assert PROFILES["noisy"].position_sigma_m > 0
    assert PROFILES["degraded"].dropout_probability > PROFILES["noisy"].dropout_probability
    assert PROFILES["degraded"].capture_delay_ticks > PROFILES["noisy"].capture_delay_ticks
