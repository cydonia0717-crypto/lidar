"""Reference-vs-optimized phase16 physical equivalence, not timing identity."""
from pathlib import Path
import csv

import pytest

from lidar_sim.experiment import RunConfig, run
from lidar_sim.phase16_equivalence import PHYSICAL_KEYS
from lidar_sim.planner import PlannerConfig


@pytest.mark.parametrize("scene,seed", [
    ("empty", 0),
    ("crossing", 3),
    ("two_targets", 1),
])
def test_phase16_short_closed_loop_matches_frozen_phase15(scene, seed):
    config = RunConfig(
        max_time=.5, goal_criterion="pose_gate",
        physical_road_footprint="rearaxle_rectangle",
        physical_body_offset_m=.55)
    original = run(scene, seed=seed, mode="two_stage_viability_phase15",
                   config=config)
    optimized = run(scene, seed=seed, mode="two_stage_viability",
                    config=config)
    assert len(original.trace) == len(optimized.trace)
    assert len(original.trace) >= 2
    for i, (a, b) in enumerate(zip(original.trace, optimized.trace)):
        for key in PHYSICAL_KEYS:
            assert a.get(key) == b.get(key), (scene, seed, i, key)
    assert original.success == optimized.success
    assert original.collision == optimized.collision
    assert original.road_exit == optimized.road_exit
    assert original.min_clearance_m == optimized.min_clearance_m


def test_phase16_reference_does_not_modify_bodyframe_mode():
    cfg = PlannerConfig()
    config = RunConfig(max_time=.3)
    before = run("empty", mode="two_stage_bodyframe", planner=cfg,
                 config=config)
    _ = run("empty", mode="two_stage_viability_phase15", planner=cfg,
            config=config)
    after = run("empty", mode="two_stage_bodyframe", planner=cfg,
                config=config)
    assert [{k:v for k,v in r.items() if k != "planning_ms"}
            for r in before.trace] == [
                {k:v for k,v in r.items() if k != "planning_ms"}
                for r in after.trace]


def test_equivalence_audit_detects_command_difference_but_ignores_cpu_noise(tmp_path):
    """The audit must fail on a control mismatch, not on timing jitter."""
    from lidar_sim.phase16_equivalence import verify_case
    old = tmp_path / "empty__nominal__0__two_stage_viability_phase15.csv"
    new = tmp_path / "empty__nominal__0__two_stage_viability.csv"

    physical = {key: "0" for key in PHYSICAL_KEYS}
    def save(path, command, ms):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[*PHYSICAL_KEYS, "planning_ms"])
            writer.writeheader()
            writer.writerow({**physical, "command_speed": command, "planning_ms": ms})

    save(old, "1.2", "80")
    save(new, "1.2", "15")
    verified = verify_case(old, new)
    assert verified["equivalent_frames"] == 1
    assert verified["speedup_ratio"] == 80 / 15
    save(new, "2.4", "15")
    with pytest.raises(AssertionError, match="first divergence"):
        verify_case(old, new)
