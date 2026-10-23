"""Replay captured planner CPU delays without sharing the original Runner load.

The lagged episode executes with exactly the submitted command timing captured
from a measured run, but the new planner's wall clock is never used for plant
execution. This is a reproducibility test, not validation on Orin hardware.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .evaluation import TrialSpec, make_trial
from .experiment import RunConfig, run
from .phase16_equivalence import PHYSICAL_KEYS

REPLAY_KEYS = (*PHYSICAL_KEYS, "compute_busy",
               "compute_submit_delay_ms", "planner_busy_until")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def recorded_delays(rows: list[dict]) -> tuple[float, ...]:
    result = []
    for r in rows:
        # Terminal and CPU-busy ticks have no freshly completed command.
        if r.get("compute_busy") != "0":
            continue
        value = r.get("compute_submit_delay_ms")
        if value in (None, ""):
            raise ValueError("decision missing actual compute submission delay")
        ms = float(value)
        if not math.isfinite(ms) or ms < 0:
            raise ValueError("invalid recorded compute delay")
        result.append(ms)
    if not result:
        raise ValueError("measured trace has no CPU delay records")
    return tuple(result)


def compare_rows(reference: list[dict], replayed: list[dict]) -> float:
    if len(reference) != len(replayed):
        raise AssertionError(
            f"frame count differs: measured={len(reference)} replay={len(replayed)}")
    max_error = 0.0
    for i, (original, repeat) in enumerate(zip(reference, replayed)):
        for key in REPLAY_KEYS:
            a, b = original.get(key, ""), repeat.get(key, "")
            # A numeric float comparison tolerates textual JSON/CSV round-trip
            # and the extra multiply/divide by 1000 in scheduling.
            if a in ("", None) or b in ("", None):
                if a not in ("", None) or b not in ("", None):
                    raise AssertionError(f"row {i} t={original['time']}: {key} missing")
                continue
            try:
                av, bv = float(a), float(b)
            except (TypeError, ValueError):
                if str(a) != str(b):
                    raise AssertionError(
                        f"row {i} t={original['time']}: {key} differs: {a} vs {b}")
            else:
                error = abs(av - bv)
                max_error = max(max_error, error)
                if not math.isclose(av, bv, abs_tol=1e-9, rel_tol=1e-12):
                    raise AssertionError(
                        f"row {i} t={original['time']}: {key} differs: {a} vs {b}")
    return max_error


def verify_measured_replay(root: Path, spec: TrialSpec, mode: str,
                           *, max_time: float = 35.0,
                           source_policy: str = "measured") -> dict:
    if source_policy != "measured" and not source_policy.startswith("fixed_"):
        raise ValueError("source_policy must be measured or fixed_<ms>ms")
    if "/" in source_policy or ".." in source_policy:
        raise ValueError("unsafe source policy")
    path = (root / spec.case_id / mode / source_policy /
            f"{spec.name}_{mode}_{spec.seed}_trace.csv")
    metrics_path = path.with_name(
        f"{spec.name}_{mode}_{spec.seed}_metrics.json")
    reference = read_rows(path)
    elapsed_ms = recorded_delays(reference)
    trial = make_trial(spec)
    repeat = run(
        spec.name, spec.seed, mode,
        vehicle=trial.vehicle, planner=trial.planner,
        world_override=trial.world,
        config=RunConfig(
            max_time=max_time, physical_road_footprint="rearaxle_rectangle",
            physical_body_offset_m=.55, goal_criterion="pose_gate",
            compute_delay_mode="replay", compute_replay_ms=elapsed_ms))
    difference = compare_rows(reference, repeat.trace)
    expected = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("success", "collision", "road_exit", "timeout",
                "missed_control_ticks", "submitted_commands"):
        if expected[key] != getattr(repeat, key):
            raise AssertionError(f"{spec.case_id}/{mode}: {key} changed")
    for key in ("duration_s", "min_clearance_m"):
        if not math.isclose(expected[key], getattr(repeat, key), abs_tol=1e-9):
            raise AssertionError(f"{spec.case_id}/{mode}: {key} changed")
    return {
        "case_id": spec.case_id, "mode": mode,
        "replayed_decisions": len(elapsed_ms),
        "matched_frames": len(reference),
        "max_state_abs_error": difference,
        "collision": repeat.collision, "success": repeat.success,
        "missed_control_ticks": repeat.missed_control_ticks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic phase17 CPU-delay replay")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenario", default="two_targets")
    parser.add_argument("--stress", default="weak_brake")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", default="two_stage_viability")
    parser.add_argument("--source-policy", default="measured",
                        help="Recorded measured or fixed_<ms>ms policy directory")
    parser.add_argument("--max-time", type=float, default=35.0)
    parser.add_argument("--output", type=Path,
                        default=Path("results/phase17-replay.csv"))
    args = parser.parse_args()
    if args.seed < 0 or args.max_time <= 0 or not math.isfinite(args.max_time):
        parser.error("invalid seed/time")
    spec = TrialSpec(args.scenario, args.seed, args.stress)
    rows = [verify_measured_replay(args.root, spec, mode.strip(),
                                  max_time=args.max_time,
                                  source_policy=args.source_policy)
            for mode in args.modes.split(",") if mode.strip()]
    if not rows:
        parser.error("empty modes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['case_id']}/{row['mode']}: "
              f"{row['matched_frames']} frames identical; "
              f"max error={row['max_state_abs_error']:.2g}, "
              f"collision={row['collision']}")


if __name__ == "__main__":
    main()
