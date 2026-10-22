import csv
import json

from lidar_sim.experiment import batch_compare, run, save_result


def test_empty_scene_reaches_goal():
    result = run("empty", seed=0)
    assert result.success
    assert not result.collision
    assert not result.road_exit
    assert len(result.trace) > 5


def test_simulation_deterministic():
    a = run("crossing", seed=3)
    b = run("crossing", seed=3)
    # Wall-clock planner timings vary even when physical rollout is deterministic.
    assert [{k: v for k, v in row.items() if k != "planning_ms"}
            for row in a.trace] == [
                {k: v for k, v in row.items() if k != "planning_ms"}
                for row in b.trace]
    assert a.min_clearance_m == b.min_clearance_m


def test_save_result(tmp_path):
    result = run("empty")
    metrics, trace = save_result(result, tmp_path)
    assert json.loads(metrics.read_text())["scenario"] == "empty"
    with trace.open(newline="") as f:
        assert len(list(csv.DictReader(f))) == len(result.trace)


def test_comparison_paired_seeds():
    records = batch_compare(("empty",), seeds=2)
    assert len(records) == 4
    assert {(r["seed"], r["mode"]) for r in records} == {(0,"reactive"),(0,"predictive"),(1,"reactive"),(1,"predictive")}
