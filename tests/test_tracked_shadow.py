"""Strict track ingestion and non-actuating planner replay tests."""
import copy
import json
from dataclasses import replace

import pytest

from lidar_sim.model import Command, VehicleConfig
from lidar_sim.tracked_shadow import (
    SnapshotError, evaluate_file, parse_snapshot, shadow_decision,
)


def snapshot(t=1.0, observed_at=.9):
    return {
        "t_s": t,
        "frame": "odom",
        "ego": {
            "x_m": 1.0, "y_m": -.1, "yaw_rad": 0.02,
            "speed_mps": 1.5, "steer_rad": 0.0,
        },
        "road": {
            "half_width_m": 3.25, "goal_x_m": 28.0,
            "safety_margin_m": .2,
        },
        "actuator": {
            "delay_s": .25,
            "active": {"target_speed_mps": 1.5, "target_steer_rad": 0},
            "pending": [
                {"activate_at_s": t+.12,
                 "target_speed_mps": .6, "target_steer_rad": .1},
                {"activate_at_s": t+.20,
                 "target_speed_mps": 0., "target_steer_rad": .0},
            ],
        },
        "tracks": [
            {"id": "tracked_person_07", "observed_at_s": observed_at,
             "x_m": 13.0, "y_m": -4.4, "vx_mps": -.2,
             "vy_mps": .58, "radius_m": .35,
             "point_count": 8, "sigma_x_m": .15, "sigma_y_m": .22},
        ],
    }


def test_track_state_is_rebased_to_absolute_observation_time():
    s = parse_snapshot(snapshot(t=1.0, observed_at=.9))
    obs = s.world.obstacles[0]
    assert obs.name == "tracked_person_07"
    assert obs.position(.9) == pytest.approx((13., -4.4))
    assert obs.position(1.0) == pytest.approx((12.98, -4.342))
    assert obs.position(1.8) == pytest.approx((12.82, -3.878))
    assert s.state.time == 1.0
    assert s.oldest_track_age_s == pytest.approx(.1)


def test_shadow_does_not_execute_planner_or_infer_empty_free_space(monkeypatch):
    import lidar_sim.tracked_shadow as shadow

    def must_not_plan(*args, **kwargs):
        raise AssertionError("planner was called on untrusted observations")

    monkeypatch.setattr(shadow, "choose_action", must_not_plan)
    empty = snapshot()
    empty["tracks"] = []
    e = shadow_decision(parse_snapshot(empty))
    assert e["status"] == "no_observed_tracks_unverified_free_space"
    assert e["recommended_speed_mps"] == ""
    stale = shadow_decision(parse_snapshot(snapshot(t=1.0, observed_at=.2)))
    assert stale["status"] == "stale_track_estimates"
    assert stale["stale_track_ids"] == "tracked_person_07"
    assert stale["recommended_speed_mps"] == ""


def test_read_only_decision_preserves_actual_actuator_queue():
    measured = parse_snapshot(snapshot())
    active = measured.actuator.active
    pending = tuple(measured.actuator.pending)
    result = shadow_decision(measured)
    assert result["status"] in (
        "evaluated_not_safety_certified",
        "unsafe_fallback_no_guaranteed_action",
    )
    assert result["shadow_only"] == 1
    assert result["mode"] == "two_stage_viability"
    assert isinstance(result["planner_wall_ms"], float)
    assert isinstance(result["recommended_speed_mps"], float)
    assert measured.actuator.active == active
    assert tuple(measured.actuator.pending) == pending


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(frame="base_link"), "frame"),
    (lambda d: d["tracks"][0].update(observed_at_s=2), "future"),
    (lambda d: d["tracks"][0].update(radius_m=0), "radius"),
    (lambda d: d["tracks"].append(dict(d["tracks"][0])), "unique"),
    (lambda d: d["tracks"][0].update(sigma_x_m=-1), "sigma_x"),
    (lambda d: d["tracks"][0].update(vx_mps=float("nan")), "finite"),
    (lambda d: d["tracks"][0].update(point_count=-1), "point_count"),
    (lambda d: d["actuator"].pop("pending"), "pending"),
    (lambda d: d["actuator"]["pending"].reverse(), "ordered"),
    (lambda d: d["actuator"]["pending"][0].update(activate_at_s=0.5), "already"),
    (lambda d: d["actuator"].update(delay_s=-.25), "delay"),
    (lambda d: d["ego"].update(speed_mps=5.), "ego speed"),
    (lambda d: d.pop("tracks"), "tracks"),
])
def test_fail_closed_for_bad_coordinates_timestamps_queues_or_quality(mutate, match):
    s = snapshot()
    mutate(s)
    with pytest.raises(SnapshotError, match=match):
        parse_snapshot(s)


def test_offline_jsonl_processes_valid_stale_empty_and_gaps(tmp_path):
    valid = snapshot(1.0, .95)
    stale = snapshot(1.1, .2)
    empty = snapshot(1.2, 1.2)
    empty["tracks"] = []
    gap = snapshot(2.0, 1.95)
    input_file = tmp_path / "tracked.jsonl"
    input_file.write_text("\n".join(json.dumps(row) for row in
                                     (valid, stale, empty, gap)) + "\n",
                          encoding="utf-8")
    rows = evaluate_file(input_file)
    assert len(rows) == 4
    assert rows[0]["status"] in (
        "evaluated_not_safety_certified",
        "unsafe_fallback_no_guaranteed_action",
    )
    assert rows[1]["status"] == "stale_track_estimates"
    assert rows[2]["status"] == "no_observed_tracks_unverified_free_space"
    assert rows[3]["status"] == "timestamp_gap_no_recommendation"
    assert all(row["shadow_only"] == 1 for row in rows)


def test_repeated_timestamp_and_missing_frame_are_errors(tmp_path):
    sample = snapshot()
    source = tmp_path / "invalid.jsonl"
    source.write_text(json.dumps(sample)+"\n"+json.dumps(sample)+"\n",
                      encoding="utf-8")
    with pytest.raises(SnapshotError, match="strictly increasing"):
        evaluate_file(source)
    sample["frame"] = "map"
    source.write_text(json.dumps(sample), encoding="utf-8")
    with pytest.raises(SnapshotError, match="line 1: frame"):
        evaluate_file(source)


def test_age_threshold_exact_boundary_is_allowed():
    d = snapshot(t=1.0, observed_at=.65)
    allowed = parse_snapshot(d, max_age_s=.35)
    assert not allowed.stale_track_ids
    disallowed = parse_snapshot(d, max_age_s=.34)
    assert disallowed.stale_track_ids


def test_empty_input_fails(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="no snapshots"):
        evaluate_file(path)
