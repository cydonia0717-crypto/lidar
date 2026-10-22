"""CI integration gate: run phase15 vs phase16 in the *same* process.

Unlike a workflow that might not be enabled for a feature branch, this test is
included in the repository's existing full pytest job. It persists artifacts
under results/ so that the existing upload-artifact step can retain them.
The timings are observational and never part of the pass/fail predicate.
"""
from pathlib import Path
import csv

from lidar_sim.evaluation import TrialSpec, evaluate_cases, save_study
from lidar_sim.phase16_equivalence import verify_root
from lidar_sim.phase16_timing import summarize
from lidar_sim.phase16_profile import profile


def test_phase16_known_critical_closed_loop_equivalence():
    root = Path("results/phase16-gate")
    cases = [TrialSpec("two_targets", seed, "weak_brake") for seed in (0, 1)]
    rows = evaluate_cases(
        cases,
        modes=("two_stage_viability_phase15", "two_stage_viability"),
        max_time=35.0,
        physical_road_footprint="rearaxle_rectangle",
        goal_criterion="pose_gate",
        assumed_body_offset_m=0.55,
        physical_body_offset_m=0.55,
        all_trace_dir=root / "all_traces",
    )
    save_study(rows, root, inputs={"goal_criterion": "pose_gate"})
    verified = verify_root(root)
    assert len(verified) == len(cases)
    # Assertions above include every issued command, applied command,
    # physical state, clearance, fallback, and episode outcome.
    assert all(r["phase15_decisions"] == r["phase16_decisions"]
               for r in verified)
    assert all(r["equivalent_frames"] > 100 for r in verified)
    # Persist the raw timing distribution summaries and equivalence verdict
    # in the existing Actions job artifact rather than only in pytest output.
    def write_rows(filename, values):
        with (root / filename).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    write_rows("phase16_equivalence.csv", verified)
    traces = sorted((root / "all_traces").glob("*.csv"))
    write_rows("phase16_timing.csv", [summarize(p) for p in traces])
    write_rows("phase16_profile.csv", [
        profile(p) for p in traces
        if p.name.endswith("__two_stage_viability.csv")
    ])

