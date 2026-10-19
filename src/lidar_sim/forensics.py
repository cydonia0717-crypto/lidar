"""Forensics for paired dual-target collision and emergency-fallback episodes.

This module diagnoses actual saved/replicated synthetic rollouts. It does not
retroactively "fix" collisions or manufacture a favorable baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .evaluation import TrialSpec, make_trial, parse_names, VALID_STRESSES, VALID_MODES
from .experiment import RunConfig, run, Result
from .world import World
from .model import VehicleConfig


def target_clearances(row: dict, world: World, car: VehicleConfig) -> list[tuple[str, float]]:
    """Signed body-obstacle gaps calculated at the actual simulated timestamp."""
    values = []
    for obstacle in world.obstacles:
        ox, oy = obstacle.position(float(row["time"]))
        d = math.hypot(float(row["x"]) - ox, float(row["y"]) - oy)
        values.append((obstacle.name, d - car.collision_radius - obstacle.radius))
    return sorted(values, key=lambda item: item[1])


def forensic_report(result: Result, world: World, vehicle: VehicleConfig,
                    *, pre_collision_steps: int = 16) -> dict:
    if pre_collision_steps < 1:
        raise ValueError("pre_collision_steps must be >=1")
    if not result.trace:
        raise ValueError("result has no trace")
    rows = result.trace
    last = rows[-1]
    hazards = []
    for row in rows:
        distances = target_clearances(row, world, vehicle)
        nearest = distances[0] if distances else ("none", float("inf"))
        if nearest[1] <= world.safety_margin + 0.6:
            hazards.append((float(row["time"]), nearest[0], nearest[1]))
    collision_frame = None
    if result.collision or result.road_exit:
        collision_frame = len(rows) - 1

    def first_step(predicate):
        hits = [r for r in rows if predicate(r)]
        return hits[0]["time"] if hits else None

    start = max(0, (collision_frame if collision_frame is not None else len(rows)-1) - pre_collision_steps)
    sampled = []
    for row in rows[start:]:
        nearest = target_clearances(row, world, vehicle)
        body_gap = nearest[0][1] if nearest else None
        status = str(row.get("fallback_strategy") or "none")
        sampled.append({
            "time": round(float(row["time"]), 3),
            "x": round(float(row["x"]), 3),
            "y": round(float(row["y"]), 3),
            "speed": round(float(row["speed"]), 3),
            "steer": round(float(row["steer"]), 3),
            "cmd_speed": row.get("command_speed", ""),
            "applied_speed": row.get("applied_speed", ""),
            "backups_rejected": row.get("backup_rejections", ""),
            "fallback": status,
            "nearest": nearest[0][0] if nearest else "none",
            "body_gap_m": round(body_gap, 4) if body_gap is not None else None,
        })
    last_clearances = target_clearances(last, world, vehicle)
    return {
        "mode": result.mode,
        "scenario": result.scenario,
        "seed": result.seed,
        "success": result.success,
        "collision": result.collision,
        "road_exit": result.road_exit,
        "timeout": result.timeout,
        "end_time": round(result.duration_s, 4),
        "impact_target": last_clearances[0][0] if result.collision and last_clearances else None,
        "impact_signed_gap_m": round(last_clearances[0][1], 4) if last_clearances else None,
        "first_proximity_time": hazards[0][0] if hazards else None,
        "first_unsafe_fallback_time": first_step(lambda r: r.get("unsafe_fallback") == 1),
        "first_backup_rejection_time": first_step(lambda r: isinstance(r.get("backup_rejections"), int)
                                               and r["backup_rejections"] > 0),
        "max_backups_rejected_in_one_cycle": max(
            (r.get("backup_rejections", 0) for r in rows
             if isinstance(r.get("backup_rejections"), int)), default=0),
        "unsafe_fallback_steps": result.unsafe_fallback_steps,
        "backup_rejections": result.backup_rejections,
        "recent_trace": sampled,
    }


def _csv_output(path: Path, reports: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        columns = [key for key in reports[0] if key != "recent_trace"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in reports)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Repeat dual-target failures with exact trajectory forensics")
    parser.add_argument("--scenario", default="two_targets")
    parser.add_argument("--stresses", default="nominal,weak_brake")
    parser.add_argument("--modes", default="delay_aware,braking")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--steps-before-end", type=int, default=12)
    parser.add_argument("--output", default="results/forensics")
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.steps_before_end < 1 or args.max_time <= 0:
        parser.error("invalid trial count, duration, or pre-end window")
    stresses = parse_names(args.stresses, VALID_STRESSES, "stresses")
    modes = parse_names(args.modes, VALID_MODES, "modes")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    reports = []
    for stress in stresses:
        for seed in range(args.seeds):
            trial = make_trial(TrialSpec(args.scenario, seed, stress))
            for mode in modes:
                result = run(args.scenario, seed, mode, vehicle=trial.vehicle,
                             planner=trial.planner, config=RunConfig(max_time=args.max_time),
                             world_override=trial.world)
                report = forensic_report(result, trial.world, trial.vehicle,
                                         pre_collision_steps=args.steps_before_end)
                report["stress"] = stress
                reports.append(report)
                status = "COLLISION" if report["collision"] else ("SUCCESS" if report["success"] else "OTHER")
                print(f'{stress}/{seed}/{mode}: {status}, t={report["end_time"]:.2f}s, '
                      f'target={report["impact_target"]}, '
                      f'first_fallback={report["first_unsafe_fallback_time"]}, '
                      f'first_backup_rejection={report["first_backup_rejection_time"]}, '
                      f'cumulative_backup_rejections={report["backup_rejections"]}',
                      flush=True)
                if report["collision"]:
                    print("  final states:", flush=True)
                    for row in report["recent_trace"][-6:]:
                        print("   ", json.dumps(row, ensure_ascii=False), flush=True)
    (folder / "forensics.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    _csv_output(folder / "forensics_summary.csv", reports)
    print(f"Saved {len(reports)} analytical reports to {folder}", flush=True)


if __name__ == "__main__":
    main()
