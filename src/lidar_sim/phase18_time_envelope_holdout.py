"""Predeclared fresh simulator-seed check of time-ramp versus no padding.

Seed 60-63 were already inspected in PR25/PR26 and MUST NOT be treated as a
blind holdout. This runner uses 90-97, chosen before examining those outcomes.
It makes one frozen comparison only: no padding vs the predeclared 0.1..0.5m
horizon-dependent ramp. No tuning based on outputs is allowed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .evaluation import TrialSpec
from .phase18_time_envelope_eval import (
    calibrate_horizons, paired, study, write_csv, HORIZONS, RAMP_M,
)


HOLDOUT_SEEDS = tuple(range(90, 98))
CALIBRATION_SEEDS = tuple(range(10, 20))
COMPARE = ("none", "time_ramp")


def evaluate(output: Path, *, max_time: float = 22.0) -> list[dict]:
    calibration_specs = [
        TrialSpec("two_targets", x, "weak_brake") for x in CALIBRATION_SEEDS]
    heldout = [
        TrialSpec("two_targets", x, "weak_brake") for x in HOLDOUT_SEEDS]
    if {s.case_id for s in calibration_specs} & {
            s.case_id for s in heldout}:
        raise ValueError("calibration and holdout case IDs must be disjoint")
    cal = calibrate_horizons(calibration_specs, ticks=161, dt=.1)
    output.mkdir(parents=True, exist_ok=True)
    (output/"calibration_by_horizon.json").write_text(json.dumps(
        cal, indent=2, ensure_ascii=False), encoding="utf-8")
    records = study(heldout, calibration=cal, output=output,
                    max_time=max_time, policies=COMPARE)
    write_csv(output/"paired.csv", paired(records, policies=COMPARE))
    (output/"inputs.json").write_text(json.dumps({
        "purpose": "frozen first inspection of simulator holdout seeds 90-97",
        "calibration_specs": [asdict(s) for s in calibration_specs],
        "holdout_specs": [asdict(s) for s in heldout],
        "comparison": list(COMPARE),
        "horizons_s": list(HORIZONS),
        "ramp_m": list(RAMP_M),
        "max_time": max_time,
        "warnings": [
            "8 new simulation random seeds in one weak-brake scenario, not real LiDAR.",
            "This holdout must not be reused to tune the same ramp and called blind.",
            "All collisions scored against physical truth, not inflated planning radii.",
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen 8-seed comparison of no radius vs time ramp")
    p.add_argument("--output", type=Path, default=Path("results/phase18-time-envelope-holdout-90-97"))
    p.add_argument("--max-time", type=float, default=22.)
    args = p.parse_args()
    records = evaluate(args.output, max_time=args.max_time)
    for profile in ("noisy_ls5", "degraded_ls5"):
        for policy in COMPARE:
            subset = [r for r in records
                      if r["sensor_profile"] == profile and r["policy"] == policy]
            print(f"{profile} {policy}: success={sum(r['success'] for r in subset)}/8 "
                  f"collision={sum(r['collision'] for r in subset)}/8 "
                  f"timeout={sum(r['timeout'] for r in subset)}/8")
    print("No estimator or envelope changes selected from these holdout results.")


if __name__ == "__main__":
    main()
