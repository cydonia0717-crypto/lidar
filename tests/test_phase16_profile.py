"""Unit tests for workload decomposition from actual per-step metrics."""
import csv

import pytest

from lidar_sim.phase16_profile import profile


def test_phase16_workload_profile_excludes_terminal_and_keeps_counters(tmp_path):
    path = tmp_path / "trace.csv"
    keys = [
        "planning_ms", "lookahead_parent_ms", "lookahead_child_ms",
        "lookahead_child_calls", "lookahead_probed", "lookahead_unprobed",
        "lookahead_early_exit", "lookahead_cached_alternative",
        "lookahead_no_viable_shortlist", "lookahead_cache_entries",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(dict(zip(keys, [100, 40, 50, 1, 1, 2, 1, 1, 0, 12])))
        writer.writerow(dict(zip(keys, [40, 30, 0, 0, 3, 0, 0, 0, 1, 8])))
        writer.writerow({key: "" for key in keys})
    result = profile(path)
    assert result["decisions"] == 2
    assert result["mean_planning_ms"] == pytest.approx(70)
    assert result["mean_parent_ms"] == pytest.approx(35)
    assert result["mean_child_ms"] == pytest.approx(25)
    assert result["child_fraction_of_total"] == pytest.approx(50 / 140)
    assert result["child_calls"] == 1
    assert result["skipped_candidate_probes"] == 2
    assert result["cached_alternative_ticks"] == 1
    assert result["mean_cache_entries"] == pytest.approx(10)


def test_missing_phase16_profile_diagnostics_is_rejected(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("planning_ms\n12\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        profile(path)
