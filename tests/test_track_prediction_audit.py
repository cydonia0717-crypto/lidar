"""Synthetic-only unit tests of H1 segment-level independent-reference audit."""
import csv
import json
import math
import sys
from pathlib import Path

import pytest

from lidar_sim.track_prediction_audit import (
    AuditError, Episode, conformal_quantile, evaluate_episode,
    load_labels, load_manifest, main, reference_at, summarize,
)
from lidar_sim.tracked_shadow import parse_snapshot


def write_episode(base: Path, name: str, *, good_error: float,
                  poor_error: float, incomplete: bool = False) -> tuple[str, str]:
    snapshots = base / f"{name}.jsonl"
    labels = base / f"{name}.csv"
    now = 10.0
    tracks = [
        {"id": "good", "observed_at_s": now,
         "x_m": 12.0, "y_m": -3.0,
         "vx_mps": 0, "vy_mps": 0.5, "radius_m": .4,
         "point_count": 15},
        {"id": "degraded", "observed_at_s": now-.2,
         "x_m": 20.0, "y_m": 1.0,
         "vx_mps": -.5, "vy_mps": 0, "radius_m": .4,
         "point_count": 4},
    ]
    snap = {
        "frame": "odom", "t_s": now,
        "ego": {"x_m": 1, "y_m": 0, "yaw_rad": 0,
                "speed_mps": 1.5, "steer_rad": 0},
        "actuator": {"delay_s": .25,
                     "active": {"target_speed_mps": 1.5,
                                "target_steer_rad": 0}, "pending": []},
        "road": {"half_width_m": 3.25, "goal_x_m": 28.0,
                 "safety_margin_m": .2},
        "tracks": tracks,
    }
    snapshots.write_text(json.dumps(snap)+"\n", encoding="utf-8")
    world = parse_snapshot(snap).world
    with labels.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame", "track_id", "time_s", "x_m", "y_m"])
        w.writeheader()
        for track, obs, error in zip(tracks, world.obstacles,
                                      (good_error, poor_error)):
            for h in (.5, 1., 2.):
                if incomplete and h == 2.0:
                    continue
                px, py = obs.position(now+h)
                w.writerow({
                    "frame": "odom", "track_id": track["id"],
                    "time_s": now+h, "x_m": px+error, "y_m": py,
                })
    return snapshots.name, labels.name


def make_manifest(tmp_path, *, n_cal=12, n_test=4):
    data = []
    for i in range(n_cal+n_test):
        is_cal = i < n_cal
        good = .10+(i % 3)*.05 if is_cal else .12
        poor = .30+(i % 3)*.05 if is_cal else .32
        snapshots, labels = write_episode(
            tmp_path, f"ep{i:03d}", good_error=good, poor_error=poor)
        data.append({
            "episode_id": f"ep{i:03d}",
            "split": "calibration" if is_cal else "test",
            "snapshot_jsonl": snapshots,
            "reference_csv": labels,
            "reference_kind": "independent_annotation",
        })
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    return path


def audit_manifest(path):
    all_rows = []
    for ep in load_manifest(path):
        all_rows.extend(evaluate_episode(
            ep, (.5, 1., 2.), 0.01,
            good_points=10, max_good_age_s=.15, max_track_age_s=.35))
    return all_rows, summarize(all_rows, .1)


def test_split_is_by_episode_and_grouped_envelopes_are_only_measured(tmp_path):
    manifest = make_manifest(tmp_path)
    episodes, summary = audit_manifest(manifest)
    assert len(episodes) == 32
    assert len(summary) == 4
    assert all(r["complete_segments"] == 1 for r in episodes)
    assert all(r["incomplete_or_stale_segments"] == 0 for r in episodes)
    grouped = {r["quality_group"]: r for r in summary
               if r["method"] == "grouped"}
    pooled = {r["quality_group"]: r for r in summary
              if r["method"] == "pooled"}
    assert grouped["good_observation"]["calibration_episode_units"] == 12
    assert grouped["degraded_or_unknown"]["calibration_episode_units"] == 12
    assert grouped["good_observation"]["radius_m"] == pytest.approx(.20)
    assert grouped["degraded_or_unknown"]["radius_m"] == pytest.approx(.40)
    assert pooled["good_observation"]["radius_m"] == pytest.approx(.40)
    assert all(r["test_complete_episode_units"] == 4 for r in summary)
    assert all(r["observed_fraction"] == 1 for r in summary)
    # These are synthetic demonstration errors, NOT real-LiDAR performance.


def test_small_calibration_set_refuses_fake_90_percent_envelope(tmp_path):
    _, summary = audit_manifest(make_manifest(tmp_path, n_cal=3, n_test=2))
    assert all(r["radius_m"] == "" for r in summary)
    assert all("insufficient_independent" in r["status"] for r in summary)
    assert all(r["observed_fraction"] == "" for r in summary)
    assert conformal_quantile([.1, .2, .3], .10) is None
    assert conformal_quantile([.1]*9, .10) == .1


def test_missing_future_reference_is_not_zero_error(tmp_path):
    a, b = write_episode(tmp_path, "missing", good_error=0, poor_error=0,
                         incomplete=True)
    e = Episode("missing", "test", tmp_path/a, tmp_path/b)
    rows = evaluate_episode(e, (.5, 1., 2.), .01,
                            good_points=10, max_good_age_s=.15,
                            max_track_age_s=.35)
    assert all(r["complete_segments"] == 0 for r in rows)
    assert all(r["incomplete_or_stale_segments"] == 1 for r in rows)
    assert all(r["worst_complete_segment_error_m"] == "" for r in rows)


def test_reference_matching_must_be_independent_and_same_frame(tmp_path):
    a, b = write_episode(tmp_path, "labels", good_error=0, poor_error=.1)
    reference = load_labels(tmp_path / b)
    assert reference_at(reference, "good", 10.5, .001) == pytest.approx(
        (12.0, -2.75))
    assert reference_at(reference, "good", 10.6, .01) is None
    assert reference_at(reference, "unknown", 10.5, 1) is None
    with (tmp_path/b).open("a", encoding="utf-8") as f:
        f.write("odom,good,10.5,12.0,-2.75\n")
    with pytest.raises(AuditError, match="duplicated"):
        load_labels(tmp_path / b)


def test_manifest_rejects_leaky_duplicate_episode_and_invalid_reference(tmp_path):
    path = make_manifest(tmp_path, n_cal=2, n_test=1)
    content = path.read_text(encoding="utf-8")
    path.write_text(content+"ep000,test,ep000.jsonl,ep000.csv,independent_annotation\n",
                    encoding="utf-8")
    with pytest.raises(AuditError, match="duplicate episode_id"):
        load_manifest(path)
    path.write_text(content.replace("independent_annotation", "online_kf_estimate"),
                    encoding="utf-8")
    with pytest.raises(AuditError, match="reference_kind"):
        load_manifest(path)


def test_epoch_timestamp_is_anchored_without_cancellation():
    stamp = 1_792_812_345.0
    track = {
        "frame": "odom", "t_s": stamp + .25,
        "ego": {"x_m": 1, "y_m": 0, "yaw_rad": 0,
                "speed_mps": 1.5, "steer_rad": 0},
        "actuator": {"delay_s": .25,
                     "active": {"target_speed_mps": 1.5,
                                "target_steer_rad": 0}, "pending": []},
        "road": {"half_width_m": 3.25, "goal_x_m": 28.0,
                 "safety_margin_m": .2},
        "tracks": [{
            "id": "target", "observed_at_s": stamp, "x_m": 12.123456789,
            "y_m": -4.123456789, "vx_mps": .34567, "vy_mps": -.45678,
            "radius_m": .3,
        }],
    }
    p = parse_snapshot(track).world.obstacles[0]
    assert p.position(stamp) == (12.123456789, -4.123456789)
    assert p.position(stamp+1) == pytest.approx(
        (12.123456789+.34567, -4.123456789-.45678), abs=1e-12)


def test_audit_cli_writes_real_summary_columns_without_claiming_lidar(tmp_path, monkeypatch):
    path = make_manifest(tmp_path)
    out = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", [
        "audit", "--manifest", str(path), "--output", str(out),
        "--reference-tolerance-s", "0.01",
    ])
    main()
    assert (out / "episode_segment_error.csv").exists()
    assert (out / "coverage_audit.csv").exists()
    per_horizon = out / "per_horizon_residuals.csv"
    assert per_horizon.exists()
    with per_horizon.open(encoding="utf-8", newline="") as f:
        scores = list(csv.DictReader(f))
    assert len(scores) == 16*2*3
    assert all(row["status"] == "matched" for row in scores)
    assert {float(row["horizon_s"]) for row in scores} == {.5, 1.0, 2.0}

    meta = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert meta["calibration_episodes"] == 12
    assert meta["test_episodes"] == 4
    assert "not verified" in meta["reference_kind"]


def test_per_horizon_missing_reference_is_explicit(tmp_path):
    a, b = write_episode(tmp_path, "missing_h", good_error=.1,
                         poor_error=.2, incomplete=True)
    rows = []
    episode = Episode("missing_h", "test", tmp_path / a, tmp_path / b)
    evaluate_episode(episode, (.5, 1., 2.), .01,
                     good_points=10, max_good_age_s=.15,
                     max_track_age_s=.35, horizon_rows=rows)
    assert len(rows) == 6
    missing = [r for r in rows if r["status"] == "missing_independent_reference"]
    assert len(missing) == 2
    assert all(r["horizon_s"] == 2.0 for r in missing)
    assert all(r["radial_error_m"] == "" for r in missing)
    assert len([r for r in rows if r["status"] == "matched"]) == 4
