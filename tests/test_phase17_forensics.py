"""Check the interpretation and fail-closed behavior of CPU-delay forensics."""
import csv
import json

import pytest

from lidar_sim.phase17_forensics import (
    first_difference, first_event, final_parked_since, audit_case, audit_root,
)


def test_timeline_detects_first_physical_change_and_parked_contact():
    a = [
        {"time": "0.0", "x": "0", "y": "0", "speed": "1", "command_speed": "2"},
        {"time": "0.1", "x": ".1", "y": "0", "speed": "1", "command_speed": "2"},
        {"time": "0.2", "x": ".2", "y": "0", "speed": "0", "command_speed": ""},
    ]
    b = [
        {"time": "0.0", "x": "0", "y": "0", "speed": "1", "command_speed": "2"},
        {"time": "0.1", "x": ".099", "y": "0", "speed": "1", "command_speed": "1"},
        {"time": "0.2", "x": ".099", "y": "0", "speed": "0", "command_speed": ""},
    ]
    assert first_difference(a, b, ("x",)) == pytest.approx(.1)
    assert first_difference(a, b, ("command_speed",)) == pytest.approx(.1)
    assert final_parked_since(b) == pytest.approx(.2)
    assert first_event(b, "unsafe_fallback") is None


def test_parked_tail_detects_long_stop_and_fails_for_motion():
    rows = [
        {"time": "0", "x": "0", "y": "0", "speed": "1"},
        {"time": "1", "x": "2", "y": "0", "speed": "0"},
        {"time": "3", "x": "2", "y": "0", "speed": "0"},
    ]
    assert final_parked_since(rows) == 1
    rows[-1]["speed"] = ".1"
    assert final_parked_since(rows) is None


def test_audit_reports_stationary_collision_only_when_supported(tmp_path):
    folder = tmp_path / "two_targets__weak_brake__0" / "two_stage_viability"
    measured = folder / "measured" / "test_trace.csv"
    none = folder / "none" / "test_trace.csv"
    for path in (measured, none):
        path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "time", "x", "y", "speed", "min_clearance",
                "command_speed", "unsafe_fallback", "compute_busy"])
            writer.writeheader()
            for t, x, v, bad in [(0, 0, 1, 0), (1, 2, 0, 1),
                                 (2, 2, 0, 1), (4, 2, 0, 0)]:
                writer.writerow(dict(time=t, x=x, y=0, speed=v,
                                     min_clearance=(-.01 if t == 4 else 1),
                                     command_speed=0, unsafe_fallback=bad,
                                     compute_busy=0))
    metrics = measured.with_name("test_metrics.json")
    metrics.write_text(json.dumps({
        "collision": True, "road_exit": False, "success": False,
        "missed_control_ticks": 0, "unsafe_fallback_steps": 2,
    }), encoding="utf-8")
    report = audit_case(none, measured)
    assert report["collision"] is True
    assert report["stationary_contact_after_2s"] is True
    assert report["parked_since_s"] == 1
    assert report["first_unsafe_fallback_s"] == 1
    assert len(audit_root(tmp_path)) == 1
