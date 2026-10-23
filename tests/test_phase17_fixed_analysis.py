"""Check fixed-lag audit against complete per-policy trace metadata."""
import csv
import json

import pytest

from lidar_sim.phase17_fixed_analysis import analyze_sweep


def write_case(root, mode, lag, *, collision=False):
    case = "two_targets__weak_brake__0"
    subdir = root / case / mode / f"fixed_{lag:g}ms"
    subdir.mkdir(parents=True, exist_ok=True)
    prefix = f"two_targets_{mode}_0"
    path = subdir / f"{prefix}_trace.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "time", "x", "y", "speed", "min_clearance",
            "compute_busy", "compute_submit_delay_ms", "unsafe_fallback",
        ])
        w.writeheader()
        w.writerow({"time": 0, "x": 0, "y": 0, "speed": 1,
                    "min_clearance": 2, "compute_busy": 0,
                    "compute_submit_delay_ms": lag, "unsafe_fallback": 0})
        w.writerow({"time": 3, "x": 1, "y": 0, "speed": 0,
                    "min_clearance": -0.1 if collision else 1,
                    "compute_busy": "", "compute_submit_delay_ms": "",
                    "unsafe_fallback": ""})
    (subdir / f"{prefix}_metrics.json").write_text(json.dumps({
        "success": not collision, "collision": collision,
        "timeout": False, "road_exit": False,
    }), encoding="utf-8")
    return {
        "case_id": case, "mode": mode, "compute_policy": "fixed",
        "fixed_compute_ms": lag,
        "success": not collision, "collision": collision,
        "timeout": False, "road_exit": False,
        "submitted_commands": 1, "min_clearance_m": -.1 if collision else 1,
        "missed_control_ticks": 0,
    }


def save_episodes(root, rows):
    with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_non_monotone_lag_outcomes_are_preserved(tmp_path):
    rows = []
    for mode in ("two_stage_bodyframe", "two_stage_viability"):
        for lag, collision in ((0, False), (40, True), (100, False),
                               (160, True)):
            rows.append(write_case(tmp_path, mode, lag,
                                   collision=collision))
    save_episodes(tmp_path, rows)
    report = analyze_sweep(tmp_path)
    assert len(report) == 8
    for mode in ("two_stage_bodyframe", "two_stage_viability"):
        subset = [r for r in report if r["mode"] == mode]
        assert [r["collision"] for r in subset] == [
            False, True, False, True]
        assert sum(r["changed_outcome_from_previous_lag"] for r in subset) == 3
        assert [r["fixed_compute_ms"] for r in subset] == [0, 40, 100, 160]


def test_mismatched_fixed_lag_grids_fail(tmp_path):
    rows = [write_case(tmp_path, "two_stage_bodyframe", 0),
            write_case(tmp_path, "two_stage_bodyframe", 40),
            write_case(tmp_path, "two_stage_viability", 0)]
    save_episodes(tmp_path, rows)
    with pytest.raises(ValueError, match="identical fixed lag"):
        analyze_sweep(tmp_path)


def test_incorrect_logged_submission_lag_is_rejected(tmp_path):
    row = write_case(tmp_path, "two_stage_viability", 40)
    save_episodes(tmp_path, [row])
    trace = (tmp_path / row["case_id"] / row["mode"] / "fixed_40ms" /
             "two_targets_two_stage_viability_0_trace.csv")
    contents = trace.read_text(encoding="utf-8")
    trace.write_text(contents.replace(",40,0", ",20,0"), encoding="utf-8")
    with pytest.raises(AssertionError, match="injected compute lag"):
        analyze_sweep(tmp_path)
