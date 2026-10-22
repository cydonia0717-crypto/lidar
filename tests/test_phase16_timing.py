"""Phase 16 timing report tests."""
import csv

import pytest

from lidar_sim.phase16_timing import percentile, summarize


def test_percentile_and_deadline_counts(tmp_path):
    path = tmp_path / "trace.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time", "planning_ms"])
        w.writeheader()
        for index, sample in enumerate([20, 80, 100, 110, 200]):
            w.writerow({"time": index / 10, "planning_ms": sample})
        w.writerow({"time": 0.5, "planning_ms": ""})
    r = summarize(path)
    assert r["decisions"] == 5
    assert r["mean_ms"] == pytest.approx(102.0)
    assert r["p95_ms"] == pytest.approx(182.0)
    assert r["p99_ms"] == pytest.approx(196.4)
    assert r["over_period_count"] == 2
    assert r["over_period_fraction"] == pytest.approx(.4)
    assert percentile([3], 99) == 3


def test_no_implicit_zero_when_trace_lacks_planning_times(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("time,speed\n0.0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing per-decision"):
        summarize(path)


def test_invalid_planning_times_fail(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("planning_ms\nnan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        summarize(path)
