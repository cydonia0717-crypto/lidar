"""Synthetic-only audit of independently labelled visible targets."""
import csv
import json
import sys

import pytest

from lidar_sim.independent_visibility_audit import (
    VisibilityError, evaluate_record, load_manifest, load_visibility,
    nearest_snapshot, summarize, main,
)


def snapshot(t=1000., tracks=None):
    if tracks is None:
        tracks = [
            {"id": "lidar_a", "observed_at_s": t-.1,
             "x_m": 10., "y_m": 0., "vx_mps": 0., "vy_mps": 0.,
             "radius_m": .35},
            {"id": "lidar_b", "observed_at_s": t-.1,
             "x_m": 30., "y_m": 0., "vx_mps": 0., "vy_mps": 0.,
             "radius_m": .35},
        ]
    return {
        "frame": "odom", "t_s": t,
        "ego": {"x_m": 0., "y_m": 0., "yaw_rad": 0.,
                "speed_mps": 1.5, "steer_rad": 0.},
        "road": {"half_width_m": 3.25, "goal_x_m": 28.,
                 "safety_margin_m": .2},
        "actuator": {
            "delay_s": .25, "active": {
                "target_speed_mps": 1.5, "target_steer_rad": 0.},
            "pending": []},
        "tracks": tracks,
    }


def make_files(root, episode="ep01", *, session="walk01", split="calibration",
               snap=None, labels=None):
    if snap is None:
        snap = snapshot()
    if labels is None:
        labels = [
            ["1000", "odom", "pedestrian_1", "visible", "10.1", "0"],
            ["1000", "odom", "pedestrian_2", "visible", "30.1", "0"],
            ["1000", "odom", "pedestrian_missing", "visible", "18", "0"],
            ["1000", "odom", "behind_wall", "occluded", "", ""],
            ["1000.3", "odom", "missed_recording", "visible", "10.0", "0"],
        ]
    path = root / f"{episode}.jsonl"
    path.write_text(json.dumps(snap)+"\n", encoding="utf-8")
    label_path = root / f"{episode}.csv"
    with label_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "frame", "object_id", "visibility", "x_m", "y_m"])
        w.writerows(labels)
    return {
        "episode_id": episode, "session_id": session, "split": split,
        "snapshot_jsonl": path.name,
        "visibility_csv": label_path.name,
        "annotation_kind": "independent_visibility",
    }


def manifest(root, rows):
    path = root / "visibility_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_visibility_audit_exposes_missed_and_missing_snapshots(tmp_path):
    entry = make_files(tmp_path)
    from lidar_sim.independent_visibility_audit import Record
    record = Record(entry["episode_id"], entry["session_id"], entry["split"],
                    tmp_path/entry["snapshot_jsonl"],
                    tmp_path/entry["visibility_csv"])
    values = evaluate_record(record)
    assert {r["object_id"]: r["status"] for r in values} == {
        "pedestrian_1": "geometrically_associated",
        "pedestrian_2": "geometrically_associated",
        "pedestrian_missing": "visible_but_no_gated_track",
        "behind_wall": "excluded_not_visible",
        "missed_recording": "no_contemporaneous_tracker_snapshot",
    }
    assert {r["object_id"]: r["associated_track_id"] for r in values
            if r["status"] == "geometrically_associated"} == {
        "pedestrian_1": "lidar_a", "pedestrian_2": "lidar_b"}
    report = summarize(values)[0]
    assert report["reference_visible_events"] == 4
    assert report["geometric_associations"] == 2
    assert report["visible_but_unmatched"] == 1
    assert report["visible_unknown_or_unscorable"] == 1
    assert report["conservative_association_fraction"] == .5
    assert report["optimistic_association_fraction"] == .75
    assert "not_certified_detection_recall" in report["interpretation"]


def test_non_unique_track_or_object_assignment_is_never_guessed(tmp_path):
    entry = make_files(
        tmp_path, snap=snapshot(tracks=[
            {"id": "one_track", "observed_at_s": 1000.,
             "x_m": 10., "y_m": 0., "vx_mps": 0., "vy_mps": 0.,
             "radius_m": .35}]),
        labels=[
            ["1000", "odom", "person_a", "visible", "9.85", "0"],
            ["1000", "odom", "person_b", "visible", "10.15", "0"],
        ])
    from lidar_sim.independent_visibility_audit import Record
    record = Record(entry["episode_id"], entry["session_id"], entry["split"],
                    tmp_path/entry["snapshot_jsonl"],
                    tmp_path/entry["visibility_csv"])
    result = evaluate_record(record)
    assert all(r["status"] == "ambiguous_object_track_association"
               for r in result)
    assert all(r["associated_track_id"] == "" for r in result)


def test_multiple_tracks_in_one_gate_is_not_false_match(tmp_path):
    entry = make_files(
        tmp_path, snap=snapshot(tracks=[
            {"id": "a", "observed_at_s": 1000.,
             "x_m": 9.8, "y_m": 0., "vx_mps": 0., "vy_mps": 0.,
             "radius_m": .35},
            {"id": "b", "observed_at_s": 1000.,
             "x_m": 10.2, "y_m": 0., "vx_mps": 0., "vy_mps": 0.,
             "radius_m": .35},
        ]),
        labels=[["1000", "odom", "person", "visible", "10", "0"]])
    from lidar_sim.independent_visibility_audit import Record
    result = evaluate_record(Record(
        entry["episode_id"], entry["session_id"], entry["split"],
        tmp_path/entry["snapshot_jsonl"], tmp_path/entry["visibility_csv"]))
    assert result[0]["status"] == "ambiguous_multiple_tracks_in_gate"


def test_manifest_rejects_same_session_leaking_to_test(tmp_path):
    rows = [
        make_files(tmp_path, "cal", session="sceneA", split="calibration"),
        make_files(tmp_path, "test", session="sceneA", split="test"),
    ]
    path = manifest(tmp_path, rows)
    with pytest.raises(VisibilityError, match="session leakage"):
        load_manifest(path)
    rows[1]["session_id"] = "sceneB"
    manifest(tmp_path, rows)
    entries = load_manifest(path)
    assert len(entries) == 2


def test_label_schema_duplicate_and_future_frame_rejected(tmp_path):
    item = make_files(tmp_path)
    labels = tmp_path/item["visibility_csv"]
    with labels.open("a", encoding="utf-8") as f:
        f.write("1000,odom,pedestrian_1,visible,10,0\n")
    with pytest.raises(VisibilityError, match="duplicate"):
        load_visibility(labels)
    make_files(tmp_path, labels=[
        ["1000", "base_link", "bad", "visible", "10", "0"]])
    with pytest.raises(VisibilityError, match="odom"):
        load_visibility(labels)


def test_timestamp_tie_is_marked_ambiguous():
    assert nearest_snapshot([10., 10.1], 10.05, .051) == -1
    assert nearest_snapshot([10., 10.1], 10.15, .051) == 1
    assert nearest_snapshot([10., 10.1], 11, .051) is None


def test_cli_writes_only_synthetic_evaluation(tmp_path, monkeypatch):
    rows = [
        make_files(tmp_path, "cal", session="sceneA", split="calibration"),
        make_files(tmp_path, "test", session="sceneB", split="test"),
    ]
    manifest_path = manifest(tmp_path, rows)
    output = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", [
        "audit", "--manifest", str(manifest_path), "--output", str(output)])
    main()
    with (output/"visibility_episode_summary.csv").open(
            newline="", encoding="utf-8") as f:
        report = list(csv.DictReader(f))
    assert len(report) == 2
    assert all(int(r["geometric_associations"]) == 2 for r in report)
    assert all(float(r["conservative_association_fraction"]) == .5
               for r in report)
    metadata = json.loads((output/"run_config.json").read_text())
    assert "not independently verified" in metadata["annotation_kind"]
