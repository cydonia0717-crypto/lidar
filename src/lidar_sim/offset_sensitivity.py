"""Reproducible rear-axle offset sensitivity and weak-braking failure forensics.

Planning offset and PHYSICAL truth are independently controlled. Every
scenario is the same world for all grid cells at a given seed/stress.
Measurements are from closed-loop synthetic CV rollouts, not real LiDAR.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import mean

from .evaluation import TrialSpec, make_trial
from .experiment import RunConfig, run
from .world import World
from .model import State, VehicleConfig


def parse_offsets(text: str) -> tuple[float, ...]:
    try:
        out = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as error:
        raise ValueError("offsets must be numeric") from error
    if not out or len(set(out)) != len(out) or any(
            not math.isfinite(v) or not 0 <= v <= VehicleConfig().length / 2
            for v in out):
        raise ValueError("offsets must be distinct and within physical body length")
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as f:
        writer=csv.DictWriter(f, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def true_gap(state: State, world: World, physical_vehicle: VehicleConfig) -> dict:
    """Signed measured gaps at one simulated instant, same physical truth."""
    road=world.road_clearance(state, physical_vehicle)-world.safety_margin
    names=[]
    for obs in world.obstacles:
        ox,oy=obs.position(state.time)
        cx,cy=physical_vehicle.body_center(state)
        names.append((math.hypot(cx-ox,cy-oy)
                      -physical_vehicle.collision_radius-obs.radius, obs.name))
    obstacle,name=min(names, default=(float("inf"),""))
    return {"physical_road_gap_m":road,
            "physical_obstacle_gap_m":obstacle,
            "nearest_obstacle":name}


def forensic_frames(trace: list[dict], world: World,
                    physical: VehicleConfig) -> list[dict]:
    frames=[]
    for r in trace:
        st=State(r["x"],r["y"],r["yaw"],r["speed"],r["steer"],r["time"])
        gap=true_gap(st,world,physical)
        frames.append({
            "time":r["time"],"x":r["x"],"y":r["y"],
            "yaw_rad":r["yaw"],"speed":r["speed"],"steer":r["steer"],
            "target_speed":r.get("command_speed"),
            "target_steer":r.get("command_steer"),
            "feasible_pairs":r.get("feasible_pairs"),
            "forward_first_available":r.get("moving_first_action_feasible"),
            "unsafe_fallback":r.get("unsafe_fallback"),
            "fallback_strategy":r.get("fallback_strategy"),
            "body_center_x":r.get("body_center_x"),
            "body_center_y":r.get("body_center_y"),
            "heading_error_deg":r.get("abs_heading_error_deg"),
            "historical_x_line_met":r.get("historical_x_line_met"),
            "pose_gate_met":r.get("pose_gate_met"),
            **{k:r.get(k) for k in r if k.startswith("reject_")},
            **gap})
    return frames


def summarize_windows(frames: list[dict], *, window_s: float=5.) -> list[dict]:
    if window_s <= 0:
        raise ValueError("window must be positive")
    if not frames:
        return []
    t0=frames[0]["time"]
    tmax=frames[-1]["time"]
    out=[]
    i=0
    while t0+i*window_s <= tmax+1e-9:
        start=t0+i*window_s
        chunk=[x for x in frames if start <= x["time"] < start+window_s
               and isinstance(x["target_speed"], (float,int))]
        if chunk:
            valid=[x["feasible_pairs"] for x in chunk
                   if isinstance(x["feasible_pairs"],(float,int))]
            row={
                "start_s":start,"end_s":start+window_s,"n":len(chunk),
                "x_progress_m":chunk[-1]["x"]-chunk[0]["x"],
                "mean_speed_mps":mean(x["speed"] for x in chunk),
                "stopped_fraction":mean(x["speed"] < .1 for x in chunk),
                "zero_command_fraction":mean(x["target_speed"] < .1 for x in chunk),
                "moving_first_available_fraction":mean(
                    bool(x["forward_first_available"]) for x in chunk),
                "mean_feasible_pairs":mean(valid) if valid else None,
                "minimum_true_obstacle_gap_m":min(x["physical_obstacle_gap_m"]
                                                  for x in chunk),
                "minimum_true_road_gap_m":min(x["physical_road_gap_m"] for x in chunk),
            }
            for phase in ("prefix","first","second","brake","hold"):
                for cause in ("road","obstacle"):
                    key=f"reject_{phase}_{cause}"
                    row[key]=sum(x.get(key) or 0 for x in chunk)
            out.append(row)
        i+=1
    return out


def outcome_row(outcome, scenario: str, stress: str,
                assumed: float, physical: float) -> dict:
    return {
        "scenario":scenario,"stress":stress,"seed":outcome.seed,
        "mode":outcome.mode,"assumed_body_offset_m":assumed,
        "physical_body_offset_m":physical,
        "success":outcome.success,"collision":outcome.collision,
        "timeout":outcome.timeout,"road_exit":outcome.road_exit,
        "duration_s":outcome.duration_s,
        "reached_historical_x_line":outcome.reached_historical_x_line,
        "reached_pose_gate":outcome.reached_pose_gate,
        "final_body_center_y_m":outcome.final_body_center_y_m,
        "final_abs_heading_error_deg":outcome.final_abs_heading_error_deg,
        "min_clearance_m":outcome.min_clearance_m,
        "mean_planning_ms":outcome.mean_planning_ms,
        "unsafe_fallback_steps":outcome.unsafe_fallback_steps,
        "backup_rejections":outcome.backup_rejections,
    }


def grid_study(*, scenarios: tuple[str, ...], stresses: tuple[str, ...],
               seeds: tuple[int, ...], physical_offsets: tuple[float, ...],
               assumed_offsets: tuple[float, ...], max_time: float,
               output: Path, forensic_seeds: tuple[int, ...]=(0,1)) -> list[dict]:
    if max_time <= 0 or not math.isfinite(max_time):
        raise ValueError("invalid simulation horizon")
    records=[]
    windows=[]
    for scenario in scenarios:
        for stress in stresses:
            for seed in seeds:
                trial=make_trial(TrialSpec(scenario,seed,stress))
                for truth in physical_offsets:
                    physical_car=replace(trial.vehicle,road_footprint="rearaxle_rectangle",
                                         rear_axle_to_body_center_m=truth)
                    for assumed in assumed_offsets:
                        planner_car=replace(trial.vehicle,
                            rear_axle_to_body_center_m=assumed)
                        result=run(scenario,seed,"two_stage_bodyframe",
                                   vehicle=planner_car, planner=trial.planner,
                                   world_override=trial.world,
                                   config=RunConfig(max_time=max_time,
                                       physical_road_footprint="rearaxle_rectangle",
                                       physical_body_offset_m=truth,
                                       goal_criterion="pose_gate"))
                        records.append(outcome_row(result,scenario,stress,
                                                   assumed,truth))
                        # Only fail cases or named debugging cases; record full
                        # physical trace for audit, never cherry-pick outcomes.
                        if seed in forensic_seeds and stress == "weak_brake":
                            prefix=f"{scenario}__{stress}__{seed}__physical_{truth:.2f}__assumed_{assumed:.2f}"
                            frames=forensic_frames(result.trace,trial.world,physical_car)
                            write_csv(output/"forensics"/f"{prefix}.csv",frames)
                            for row in summarize_windows(frames):
                                windows.append({"case_id":prefix,**row})
                        print(f"{scenario}/{stress}/{seed}: physical={truth:.2f} "
                              f"assumed={assumed:.2f} success={result.success} "
                              f"collision={result.collision} timeout={result.timeout} "
                              f"road_exit={result.road_exit} t={result.duration_s:.2f}s",
                              flush=True)
    write_csv(output/"episodes.csv",records)
    write_csv(output/"forensic_windows.csv",windows)
    summaries=[]
    for truth in physical_offsets:
        for assumed in assumed_offsets:
            group=[r for r in records if r["physical_body_offset_m"]==truth
                   and r["assumed_body_offset_m"]==assumed]
            if group:
                summaries.append({
                    "physical_body_offset_m":truth,
                    "assumed_body_offset_m":assumed,
                    "count":len(group),
                    "successes":sum(r["success"] for r in group),
                    "collisions":sum(r["collision"] for r in group),
                    "timeouts":sum(r["timeout"] for r in group),
                    "road_exits":sum(r["road_exit"] for r in group),
                    "mean_planning_ms":mean(r["mean_planning_ms"] for r in group),
                })
    write_csv(output/"offset_grid_summary.csv",summaries)
    (output/"design.json").write_text(json.dumps({
        "scenarios":scenarios,"stresses":stresses,"seeds":seeds,
        "physical_offsets_m":physical_offsets,
        "assumed_offsets_m":assumed_offsets,
        "mode":"two_stage_bodyframe","true_geometry":"rearaxle_rectangle",
        "goal_criterion":"pose_gate","goal_lateral_tolerance_m":.75,
        "goal_heading_tolerance_deg":15,
        "max_time_s":max_time,
        "interpretation":"synthetic ideal CV truth, no real robot or lidar calibration"
    },indent=2), encoding="utf8")
    print("summary:",str(output/"offset_grid_summary.csv"),flush=True)
    return records


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(description="Physical vs assumed rear-axle offset sensitivity")
    parser.add_argument("--scenarios",default="two_targets")
    parser.add_argument("--stresses",default="weak_brake")
    parser.add_argument("--seeds",default="0,1")
    parser.add_argument("--physical-offsets",default=".40,.55,.70")
    parser.add_argument("--assumed-offsets",default=".40,.55,.70")
    parser.add_argument("--max-time",type=float,default=30.)
    parser.add_argument("--forensic-seeds",default="0,1")
    parser.add_argument("--output",default="results/offset-sensitivity")
    args=parser.parse_args(argv)
    grid_study(
        scenarios=tuple(x.strip() for x in args.scenarios.split(",")),
        stresses=tuple(x.strip() for x in args.stresses.split(",")),
        seeds=tuple(int(x) for x in args.seeds.split(",")),
        physical_offsets=parse_offsets(args.physical_offsets),
        assumed_offsets=parse_offsets(args.assumed_offsets),
        max_time=args.max_time,output=Path(args.output),
        forensic_seeds=tuple(int(x) for x in args.forensic_seeds.split(",") if x.strip()))


if __name__=="__main__":
    main()
